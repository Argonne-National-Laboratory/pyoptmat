"""
    Module defining the key objects and functions to integrate ODEs
    and provide the sensitivity of the results with respect to the
    model parameters either through backpropogation AD or the adjoint method.

    The key functions are :py:func:`pyoptmat.odeint` and
    :py:func:`pyoptmat.odeint_adjoint`.  These functions both
    integrate a system of ODEs forward in time using one of several methods
    and have identical function signature.  The only difference is that
    backward passes over the results of :py:func:`pyoptmat.ode.odeint` will
    use PyTorch backpropogation AD to return the gradients while
    :py:func:`pyoptmat.ode.odeint_adjoint` will use the adjoint method.

    The two different backward pass options are seamless, the end-user
    can interchange them freely with other PyTorch code stacking on top
    of the integrated results and obtain gradient information just as if they
    were using AD.

    In all realistic cases explored so far the adjoint method produces faster
    results with far less memory use and should be preferred versus
    the AD variant.

    The methods currently available to solve ODEs are detailed below, but
    as a summary:

    * "forward_euler": simple forward Euler explicit integration
    * "backward_euler": simple backward Euler implicit integration

    Both options take a `block_size` parameter, which sets up
    vectorized/parallelized time integration.  This means the execution
    device integrates `block_size` time steps at once, rather than
    one at a time.  This *greatly* accelerates time integration, particularly
    on GPUs.  The optimal `block_size` parameter will vary based on your
    system and set of ODEs, but we highly recommend determining an
    optimal block size for your system and not leaving it set to the
    default value of 1.
"""
import torch

from pyoptmat import chunktime


class BackwardEulerScheme:
    """
    Integration with the backward Euler method
    """

    def form_operators(self, dy, yd, yJ, dt):
        """
        Form the residual and sparse Jacobian of the batched system

        Args:
            dy (torch.tensor): (ntime+1, nbatch, nsize) tensor with the increments in y
            yd (torch.tensor): (ntime, nbatch, nsize) tensor giving the ODE rate
            yJ (torch.tensor): (ntime, nbatch, nsize, nsize) tensor giving the derivative
                of the ODE
            dt (torch.tensor): (ntime, nbatch) tensor giving the time step

        Returns:
            R (torch.tensor): residual tensor of shape
                (nbatch, ntime * nsize)
            J (tensor.tensor): sparse Jacobian tensor of logical
                size (nbatch, ntime*nsize, ntime*nsize)
        """
        # Basic shape info
        ntime = dy.shape[0] - 1
        prob_size = dy.shape[-1]
        batch_size = dy.shape[-2]

        # Form the overall residual
        # R = dy_k - dy_{k-1} - ydot(y) * dt
        # However, to unsqueeze it we need to combine the 0th and the 2nd dimension
        R = (
            (dy[1:] - dy[:-1] - yd[1:] * dt.unsqueeze(-1))
            .transpose(0, 1)
            .flatten(start_dim=1)
        )

        # Form the overall jacobian
        # This has I-J blocks on the main block diagonal and -I on the -1 block diagonal
        I = torch.eye(prob_size, device=dt.device).expand(ntime, batch_size, -1, -1)

        return R, chunktime.BidiagonalForwardOperator(
            I - yJ[1:] * dt.unsqueeze(-1).unsqueeze(-1), -I[1:]
        )

    def update_adjoint(self, dt, J, a_prev, grads):
        """
        Update the adjoint for a block of time

        Args:
            dt (torch.tensor):      block of time steps
            J (torch.tensor):       block of Jacobians
            a_prev (torch.tensor):  previous adjoint
            grads (torch.tensor):   block of gradient values

        Returns:
            adjoint_block (torch.tensor): block of updated adjoint values
        """
        ntime = J.shape[0] - 1
        prob_size = J.shape[2]
        batch_size = J.shape[1]

        adjoint_block = torch.zeros(J.shape[:-1], dtype=J.dtype, device=J.device)
        adjoint_block[0] = a_prev

        # Invert J all at once
        I = torch.eye(prob_size, device=dt.device).expand(ntime, batch_size, -1, -1)
        lu, pivot, _ = torch.linalg.lu_factor_ex(
            I + J[:-1].transpose(-1, -2) * dt.unsqueeze(-1).unsqueeze(-1)
        )

        # This has to be done sequentially...
        for i in range(ntime):
            adjoint_block[i + 1] = torch.linalg.lu_solve(
                lu[i], pivot[i], adjoint_block[i].unsqueeze(-1)
            ).squeeze(-1)
            adjoint_block[i + 1] += grads[i + 1]

        return adjoint_block

    def accumulate(self, prev, time, y, a, grad, grad_fn):
        """
        Calculate the accumulated value given the results at each time step

        Args:
            prev (tuple of tensors): previous results
            times (tensor): (ntime+1, nbatch) tensor of times
            y (tensor): (ntime+1, nbatch, nsize) tensor of state
            a (tensor): (ntime+1, nbatch, nsize) tensor of adjoints
            grad (tensor): (ntime+1, nbatch, nsize) tensor of gradient values
            grad_fn (function): function that takes t,y,a and returns the dot product with
                the model parameters

        Returns:
            next (tuple of tensor): updated gradients
        """
        dt = time.diff(dim=0)
        g = grad_fn(time[:-1], y[:-1], (a[1:] - grad[1:]) * -dt.unsqueeze(-1))
        return tuple(pi + gi for pi, gi in zip(prev, g))


class ForwardEulerScheme:
    """
    Integration with the forward Euler method
    """

    def form_operators(self, dy, yd, yJ, dt):
        """
        Form the residual and sparse Jacobian of the batched system

        Args:
            dy (torch.tensor): (ntime+1, nbatch, nsize) tensor with the increments in y
            yd (torch.tensor): (ntime, nbatch, nsize) tensor giving the ODE rate
            yJ (torch.tensor): (ntime, nbatch, nsize, nsize) tensor giving the derivative
                of the ODE
            dt (torch.tensor): (ntime, nbatch) tensor giving the time step

        Returns:
            R (torch.tensor): residual tensor of shape
                (nbatch, ntime * nsize)
            J (tensor.tensor): sparse Jacobian tensor of logical size
                (nbatch, ntime*nsize, ntime*nsize)
        """
        # Basic shape info
        ntime = dy.shape[0] - 1
        prob_size = dy.shape[-1]
        batch_size = dy.shape[-2]

        # Form the overall residual
        # R = dy_k - dy_{k-1} - ydot(y) * dt
        # However, to unsqueeze it we need to combine the 0th and the 2nd dimension
        R = (
            (dy[1:] - dy[:-1] - yd[:-1] * dt.unsqueeze(-1))
            .transpose(0, 1)
            .flatten(start_dim=1)
        )

        # Form the overall jacobian
        # This has I on the diagonal and -I - J*dt on the off diagonal
        I = torch.eye(prob_size, device=dt.device).expand(ntime, batch_size, -1, -1)

        return R, chunktime.BidiagonalForwardOperator(
            I, -I[1:] - yJ[1:-1] * dt[1:].unsqueeze(-1).unsqueeze(-1)
        )

    def update_adjoint(self, dt, J, a_prev, grads):
        """
        Update the adjoint for a block of time

        Args:
            dt (torch.tensor):      block of time steps
            J (torch.tensor):       block of Jacobians
            a_prev (torch.tensor):  previous adjoint
            grads (torch.tensor):   block of gradient values

        Returns:
            adjoint_block (torch.tensor): block of updated adjoint values
        """
        ntime = J.shape[0] - 1

        adjoint_block = torch.zeros(J.shape[:-1], dtype=J.dtype, device=J.device)
        adjoint_block[0] = a_prev

        # This has to be done sequentially...
        for i in range(ntime):
            adjoint_block[i + 1] = adjoint_block[i] - torch.bmm(
                J[i + 1].transpose(-1, -2), adjoint_block[i].unsqueeze(-1)
            ).squeeze(-1) * dt[i].unsqueeze(-1)
            adjoint_block[i + 1] += grads[i + 1]

        return adjoint_block

    def accumulate(self, prev, time, y, a, grad, grad_fn):
        """
        Calculate the accumulated value given the results at each time step

        Args:
            prev (tuple of tensors): previous results
            times (tensor): (ntime+1, nbatch) tensor of times
            y (tensor): (ntime+1, nbatch, nsize) tensor of state
            a (tensor): (ntime+1, nbatch, nsize) tensor of adjoints
            grad (tensor): (ntime+1, nbatch, nsize) tensor of gradient values
            grad_fn (function): function that takes t,y,a and returns the dot product with
                the model parameters

        Returns:
            next (tuple of tensor): updated gradients
        """
        dt = time.diff(dim=0)
        g = grad_fn(time[1:], y[1:], a[:-1] * -dt.unsqueeze(-1))
        return tuple(pi + gi for pi, gi in zip(prev, g))


class FixedGridBlockSolver:
    """
    Parent class of solvers which operate on a fixed grid (i.e. non-adaptive methods)

    Args:
        func (function): function returning the time rate of change and the jacobian
        y0 (torch.tensor): initial condition

    Keyword Args:
        scheme (TimeIntegrationScheme):  time integration scheme, default is
            backward euler
        block_size (int): target block size
        rtol (float): relative tolerance for Newton's method
        atol (float): absolute tolerance for Newton's method
        miter (int): maximum number of Newton iterations
        sparse_linear_solver (str): method to solve batched sparse Ax = b, options
            are currently "direct" or "dense"
        adjoint_params: parameters to track for the adjoint backward pass
        guess_type (string): strategy for initial guess, options are "zero" and "previous"
    """

    def __init__(
        self,
        func,
        y0,
        scheme=BackwardEulerScheme(),
        block_size=1,
        rtol=1.0e-6,
        atol=1.0e-4,
        miter=100,
        linear_solve_method="direct",
        adjoint_params=None,
        guess_type="zero",
        **kwargs,
    ):
        # Store basic info about the system
        self.func = func
        self.y0 = y0

        # Time integration scheme
        self.scheme = scheme

        # Size information
        self.batch_size = self.y0.shape[0]
        self.prob_size = self.y0.shape[1]
        self.n = block_size

        # Solver params
        self.rtol = rtol
        self.atol = atol
        self.miter = miter

        # Store for later
        self.adjoint_params = adjoint_params

        # Setup the linear solver context
        self.linear_solve_context = chunktime.ChunkTimeOperatorSolverContext(
            linear_solve_method, **kwargs
        )

        # Initial guess for integration
        self.guess_type = guess_type

        # Cached solutions
        self.t = None
        self.result = None

    def integrate(self, t, cache_adjoint=False):
        """
        Main method: actually integrate through the times :code:`t`.

        Args:
          t (torch.tensor):       timesteps to report results at
          n (int):                target time block size

        Keyword Args:
          cache_adjoint (bool):   store the info we'll need for the
                                  adjoint pass, default `False`

        Returns:
          torch.tensor:       integrated results at times :code:`t`
        """
        result = torch.empty(
            t.shape[0], *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device
        )
        result[0] = self.y0

        for k in range(1, t.shape[0], self.n):
            result[k : k + self.n] = self.block_update(
                t[k : k + self.n],
                t[k - 1],
                result[k - 1],
                self.func,
                self._initial_guess(result, k),
            )

        # Store for the backward pass, if we're going to do that
        if cache_adjoint:
            self.t = t.flip(0)
            self.result = result.flip(0)

        return result

    def _initial_guess(self, result, k):
        """
        Form the initial guess

        Args:
            result (torch.tensor): currently-populated results
            k (int): current time step
        """
        if self.guess_type == "zero":
            guess = torch.zeros_like(result[k : k + self.n])
        elif self.guess_type == "previous":
            if k - self.n - 1 < 0:
                guess = torch.zeros_like(result[k : k + self.n])
            else:
                guess = result[(k - self.n) : k] - result[k - self.n - 1].unsqueeze(0)
            blk = self.n - result[k : k + self.n].shape[0]
            guess = guess[blk:]
        else:
            raise ValueError(f"Unknown initial guess strategy {self.guess_type}!")

        return guess.transpose(0, 1).flatten(start_dim=1)

    def _get_param_partial(self, t, y, a):
        """
        Get the parameter partial at each time in the block

        Args:
            t (torch.tensor): (nstep+1,nbatch) tensor of times
            y (torch.tensor): (nstep+1,nbatch,nsize) tensor of state
            a (torch.tensor): (nstep+1,nbatch,nsize) adjoint tensor
        """
        with torch.enable_grad():
            ydot = self.func(t, y)[0]
            return torch.autograd.grad(ydot, self.adjoint_params, a)

    def rewind(self, output_grad):
        """
        Rewind the adjoint to provide the dot product with each output_grad

        Args:
            output_grad (torch.tensor): dot product tensor
        """
        # Setup results gradients
        grad_result = tuple(
            torch.zeros(p.shape, device=output_grad.device) for p in self.adjoint_params
        )

        # Flip the output_grad
        output_grad = output_grad.flip(0)

        # Calculate starts at least gradient
        prev_adjoint = output_grad[0]

        for k in range(1, self.t.shape[0], self.n):
            # Could also cache these of course
            _, J = self.func(
                self.t[k - 1 : k + self.n], self.result[k - 1 : k + self.n]
            )

            full_adjoint = self.scheme.update_adjoint(
                self.t[k - 1 : k + self.n].diff(dim=0),
                J,
                prev_adjoint,
                output_grad[k - 1 : k + self.n],
            )

            # Ugh, best way I can think to do this is to combine everything...
            grad_result = self.scheme.accumulate(
                grad_result,
                self.t[k - 1 : k + self.n],
                self.result[k - 1 : k + self.n],
                full_adjoint,
                output_grad[k - 1 : k + self.n],
                self._get_param_partial,
            )

            # Update previous adjoint
            prev_adjoint = full_adjoint[-1]

        return grad_result

    def block_update(self, t, t_start, y_start, func, y_guess):
        """
        Solve a block of times all at once with backward Euler

        Args:
            t (torch.tensor):       block of times
            t_start (torch.tensor): time at start of block
            y_start (torch.tensor): start of block
            func (torch.nn.Module): function to use
        """
        # Various useful sizes
        n = t.shape[0]  # Number of time steps to do at once

        def RJ(dy):
            # Make things into a more rational shape
            dy = dy.reshape(self.batch_size, n, self.prob_size).transpose(0, 1)
            # Add a zero to the first dimension of dy
            dy = torch.vstack((torch.zeros_like(y_start).unsqueeze(0), dy))
            # Get actual values of state
            y = dy + y_start.unsqueeze(0).expand(n + 1, -1, -1)

            # Calculate the time steps
            times = torch.vstack((t_start.unsqueeze(0), t))
            dt = times.diff(dim=0)

            # Batch update the rate and jacobian
            yd, yJ = func(times, y)

            return self.scheme.form_operators(dy, yd, yJ, dt)

        dy = chunktime.newton_raphson_chunk(
            RJ,
            y_guess,
            self.linear_solve_context,
            rtol=self.rtol,
            atol=self.atol,
            miter=self.miter,
        )

        return dy.reshape(self.batch_size, n, self.prob_size).transpose(
            0, 1
        ) + y_start.unsqueeze(0).expand(n, -1, -1)


# Available solver methods mapped to the objects
int_methods = {
    "backward-euler": BackwardEulerScheme(),
    "forward-euler": ForwardEulerScheme(),
}


def odeint(func, y0, times, method="backward-euler", extra_params=None, **kwargs):
    """
    Solve the ordinary differential equation defined by the function :code:`func` in a way
    that will provide results capable of being differentiated with PyTorch's AD.

    The function :code:`func` is typically a PyTorch module.  It takes the time and
    state and returns the time rate of change and (optionally) the Jacobian at that
    time and state.  Not all integration methods require the Jacobian, but it's generally
    a good idea to provide it make use of the implicit integration methods.

    Input variables and initial condition have shape :code:`(nbatch, nvar)`

    Input times have shape :code:`(ntime, nbatch)`

    Output history has shape :code:`(ntime, nbatch, nvar)`

    Args:
      func (function):      returns tensor defining the derivative y_dot and,
                            optionally, the jacobian as a second return value
      y0 (torch.tensor):    initial conditions
      times (torch.tensor): time locations to provide solutions at

    Keyword Args:
      method (string):                      integration method, currently `"backward-euler"` or
                                            `"forward-euler"`
      extra-params (list of parameters):    not used here, just maintains compatibility with
                                            the adjoint interface
      kwargs:                               keyword arguments passed on to specific
                                            solver methods
    """
    solver = FixedGridBlockSolver(func, y0, scheme=int_methods[method], **kwargs)

    return solver.integrate(times)


class IntegrateWithAdjoint(torch.autograd.Function):
    # pylint: disable=abstract-method,arguments-differ
    """
    Wrapper to convince the thing that it needs to use the adjoint sensitivity
    instead of AD
    """

    @staticmethod
    def forward(ctx, solver, times, *params):
        """
        Args:
          ctx:        context object we can use to stash state
          solver:     ODE Solver object to use
          times:      times to provide output for
        """
        with torch.no_grad():
            # Do our first pass and get full results
            y = solver.integrate(times, cache_adjoint=True)
            # Save the info we will need for the backward pass
            ctx.solver = solver
            return y

    @staticmethod
    def backward(ctx, output_grad):
        """
        Args:
          ctx:            context object with state
          output_grad:    grads with which to dot product
        """
        with torch.no_grad():
            grad_tuple = ctx.solver.rewind(output_grad)
            return (None, None, *grad_tuple)


def odeint_adjoint(
    func, y0, times, method="backward-euler", extra_params=None, **kwargs
):
    """
    Solve the ordinary differential equation defined by the function :code:`func` in a way
    that will provide gradients using the adjoint trick

    The function :code:`func` is typically a PyTorch module.  It takes the time and
    state and returns the time rate of change and (optionally) the Jacobian at that
    time and state.  Not all integration methods require the Jacobian, but it's generally
    a good idea to provide it make use of the implicit integration methods.

    Input variables and initial condition have shape :code:`(nbatch, nvar)`

    Input times have shape :code:`(ntime, nbatch)`

    Output history has shape :code:`(ntime, nbatch, nvar)`

    Args:
      func (function):      returns tensor defining the derivative y_dot and,
                            optionally, the jacobian as a second return value
      y0 (torch.tensor):    initial conditions
      times (torch.tensor): time locations to provide solutions at

    Keyword Args:
      method (string):                      integration method, currently `"backward-euler"` or
                                            `"forward-euler"`
      extra-params (list of parameters):    additional parameters that need to be included
                                            in the backward pass that are not determinable
                                            via introsection of :code:`func`
      kwargs:                               keyword arguments passed on to specific
                                            solver methods
    """
    # Grab parameters for backward
    if extra_params is None:
        extra_params = []
    adjoint_params = tuple(p for p in func.parameters()) + tuple(extra_params)

    solver = FixedGridBlockSolver(
        func, y0, scheme=int_methods[method], adjoint_params=adjoint_params, **kwargs
    )

    wrapper = IntegrateWithAdjoint()

    return wrapper.apply(solver, times, *adjoint_params)
