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

  The current methods all accept a :code:`substep` option which will
  subdivide the provided time intervals into some number of subdivisions
  to decrease integration error.
"""

import torch

from pyoptmat import solvers, utility, spsolve


def linear(t0, t1, y0, y1, t):
    """
    Helper function for linear interpolation between two points

    .. math::

      z = \\frac{y_1-y_0}{t_1-t_0} \\left( t - t_0 \\right)

    Args:
      t0 (torch.tensor):     first "x" point
      t1 (torch.tensor):     second "x" point
      y0 (torch.tensor):     first "y" point
      y1 (torch.tensor):     second "y" point
      t (torch.tensor):      target "x" point

    Returns:
      torch.tensor:         interpolated values
    """
    n = y0.dim() - 1
    return (
        y0
        + (y1 - y0) / (t1 - t0)[(...,) + (None,) * n] * (t - t0)[(...,) + (None,) * n]
    )


class BlockSolver:
    """
    Will merge into below...
    
    Args:
      func (function):      function returning the time rate of change and,
                            optionally, the jacobian
      y0 (torch.tensor):    initial condition

    Keyword Args:
      n (int):              target block size
      sparse_linear_solver: method to solve batched sparse Ax = b

    """
    def __init__(self, func, y0, block_size = 1, sparse_linear_solver = spsolve.dense_solve):
        # Store basic info about the system
        self.func = func
        self.y0 = y0
        self.n = block_size
        self.solver = sparse_linear_solver

        self.batch_size = self.y0.shape[0]
        self.prob_size = self.y0.shape[1]

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
        result = torch.empty(t.shape[0], *self.y0.shape, dtype = self.y0.dtype,
                device = self.y0.device)
        result[0] = self.y0

        for k in range(1, t.shape[0], self.n):
            result[k:k+self.n] = self.block_update(t[k:k+self.n], t[k-1], result[k-1])

        return result

    def block_update(self, t, t_start, y_start):
        """
        Solve a block of times all at once with backward Euler

        Args:
            t (torch.tensor):       block of times
            t_start (torch.tensor): time at start of block
            y_start (torch.tensor): start of block
        """
        # Various useful sizes
        n = t.shape[0] # Number of time steps to do at once
        b = n * self.batch_size # Size of megabatch
        k = n * self.prob_size # Size of operators

        # Guess of zeros, why not
        y_guess = torch.zeros(self.batch_size, k, 
                dtype = t.dtype, device = t.device)

        def RJ(dy):
            # Make things into a more rational shape
            dy = dy.reshape(self.batch_size, n, self.prob_size).transpose(0,1)
            # Add a zero to the first dimension of dy
            dy = torch.vstack((torch.zeros_like(y_start).unsqueeze(0), dy))
            # Get actual values of state
            y = dy[1:] + y_start.unsqueeze(0).expand(n,-1,-1)

            # Calculate the time steps
            dt = torch.vstack((t_start.unsqueeze(0), t)).diff(dim = 0)

            # Batch update the rate and jacobian
            yd, yJ = self.func(t, y)

            # Multiply by the time step
            yd *= dt.unsqueeze(-1)
            yJ *= dt.unsqueeze(-1).unsqueeze(-1)
            
            # Form the overall residual
            # R = dy_k - dy_{k-1} - ydot(y) * dt
            # However, to unsqueeze it we need to combine the 0th and the 2nd dimension
            R = (dy[1:] - dy[:-1] - yd).transpose(0,1).flatten(start_dim=1)

            # Form the overall jacobian
            # This has I-J blocks on the main block diagonal and -I on the -1 block diagonal
            I = torch.eye(self.prob_size, device = t.device).expand(n,self.batch_size,-1,-1)
            J = spsolve.SquareBatchedBlockDiagonalMatrix([I - yJ, -I[1:]], [0, -1])

            return R, J
        
        dy = spsolve.newton_raphson_sparse(RJ, y_guess,
                                           solver = self.solver)[0].reshape(self.batch_size, n, self.prob_size).transpose(0,1)

        return dy + y_start.unsqueeze(0).expand(n,-1,-1)

class FixedGridSolver:
    """
    Superclass of all solvers that use a fixed grid (versus an adaptive method)

    Args:
      func (function):      function returning the time rate of change and,
                            optionally, the jacobian
      y0 (torch.tensor):    initial condition

    Keyword Args:
      adjoint_params        parameters to track for the adjoint backward pass
      substep (int):        subdivide each provided timestep into some number
                            of subdivisions for integration.  Default is `None`
                            (i.e. no subdivision)
      jit_mode (bool):      if true do various dangerous things to fix the
                            model structure.  Default is :code:`False`
    """

    def __init__(self, func, y0, adjoint_params=None, **kwargs):
        # Store basic info about the system
        self.func = func
        self.y0 = y0

        self.batch_size = self.y0.shape[0]
        self.prob_size = self.y0.shape[1]

        self.substep = kwargs.pop("substep", None)
        self.jit_mode = kwargs.pop("jit_mode", False)

        # Store for later
        self.adjoint_params = adjoint_params

        # Sort out if the function is providing the jacobian
        self.has_jac = True

        # Whether the method uses the current or previous step
        self.use_curr = False

        # Cached information for backward pass
        self.full_result = []
        self.full_jacobian = []
        self.full_times = []

        if not self.jit_mode:
            fake_time = torch.zeros(self.batch_size, device=y0.device)
            try:
                a, b = self.func(fake_time, self.y0)
            except ValueError:
                self.has_jac = False
            if not self.has_jac:
                a = self.func(fake_time, self.y0)

            if a.dim() != 2:
                raise ValueError(
                    "ODE solvers require batched functions returning (nbatch, nvars)!"
                )

            if self.has_jac and ((a.shape + a.shape[1:]) != b.shape):
                raise ValueError("Function returns Jacobian of the wrong shape!")

            if not self.has_jac:
                raise ValueError("This implementation requires a hard coded Jacobian!")

    def _get_param_partial(self, t, y, og):
        """
        Get the partial derivative of the state with respect to the parameters
        in the adjoint_params at an arbitrary point, dotted with the
        output_gradient

        Args:
          t (torch.tensor):      time you want
          y (torch.tensor):      state you want
          og (torch.tensor):     thing to dot with

        Returns:
          torch.tensor:           derivative dot product
        """
        with torch.enable_grad():
            ydot = self.func(t, y)[0]
            # Retain graph seems to be needed for solvers like L-BFGS
            # which go twice through the function
            return torch.autograd.grad(ydot, self.adjoint_params, og, retain_graph=True)

    def _construct_grid(self, t):
        """
        Construct the subidivided grid for substepped problems.

        The subdivided grid adds :code:`self.substep` extra steps
        between each point in the original grid :code:`t`.

        Args:
          t (torch.tensor):   initial timesteps

        Returns:
          torch.tensor:       subdivided grid
        """
        if self.substep and self.substep > 1:
            nshape = list(t.shape)
            nshape[0] = (nshape[0] - 1) * self.substep

            grid = torch.empty(nshape, device=t.device)
            incs = torch.linspace(0, 1, self.substep + 1, device=t.device)[:-1]

            i = 0
            for t1, t2 in zip(t[:-1], t[1:]):
                grid[i : i + self.substep] = (
                    incs * (t2 - t1).unsqueeze(-1) + t1.unsqueeze(-1)
                ).T
                i += self.substep

            grid[-1] = t[-1]

            return grid

        return t

    def integrate(self, t, cache_adjoint=False):
        """
        Main method: actually integrate through the times :code:`t`.

        Args:
          t (torch.tensor):       timesteps to report results at

        Keyword Args:
          cache_adjoint (bool):   store the info we'll need for the
                                  adjoint pass, default `False`

        Returns:
          torch.tensor:       integrated results at times :code:`t`
        """
        # Basic error checking
        if not self.jit_mode:
            if t.dim() != 2 or t.shape[1] != self.batch_size:
                raise ValueError("Expected times to be a ntime x batch_size array!")

        # Construct the substepped grid
        times = self._construct_grid(t)

        # Setup "real" results
        result = torch.empty(
            t.shape[0], *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device
        )
        result[0] = self.y0

        # Setup "cached" results, if required
        if cache_adjoint:
            self.full_result = torch.empty(
                times.shape[0],
                *self.y0.shape,
                dtype=self.y0.dtype,
                device=self.y0.device,
            )
            self.full_result[0] = self.y0
            self.full_jacobian = torch.empty(
                times.shape[0],
                self.batch_size,
                self.prob_size,
                self.prob_size,
                dtype=self.y0.dtype,
                device=self.y0.device,
            )
            self.full_times = times

        # Start integrating!
        y0 = self.y0
        j = 1
        for k, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
            y1, J = self._step(t0, t1, t1 - t0, y0)

            # Save the full set of results and the Jacobian, if required for backward pass
            if cache_adjoint:
                self.full_result[k + 1] = y1
                self.full_jacobian[k + 1] = J

            # Interpolate to results point, if requested
            if torch.any(t1 >= t[j]):  # This could be slow
                result[j] = linear(t0, t1, y0, y1, t[j])
                j += 1

            y0 = y1

        result[-1] = y1  # There may be a more elegant way of doing this...

        return result

    def rewind_adjoint(self, times, output_grad):
        """
        Rewind the solve the get the partials via the adjoint method, dotted
        with the :code:`output_grad`

        Args:
          times (torch.tensor):           output times required by the solver
          output_grad (torch.tensor):    dot product tensor
        """
        # Start from *end* of the output_grad -- going backwards
        l0 = output_grad[-1]
        # Useful for indexing
        nt = self.full_times.shape[0]
        # Gradient results
        grad_result = tuple(
            torch.zeros(p.shape, device=times.device) for p in self.adjoint_params
        )

        # Target output time
        j = times.shape[0] - 2

        # Run *backwards* through time
        for curr in range(nt - 2, -1, -1):
            # Setup all the state so I don't get confused
            last = curr + 1  # Very confusing lol
            tcurr = self.full_times[curr]
            tlast = self.full_times[last]
            dt = tlast - tcurr  # I define this to be positive
            # Take the adjoint step
            l1 = self._adjoint_update(dt, self.full_jacobian[last], l0)

            # Accumulate the gradients
            # The l0, l1 thing confuses me immensely, but is in fact correct
            # Note where the dt goes, this is to avoid branching later when summing
            if self.use_curr:
                p = self._get_param_partial(
                    tcurr, self.full_result[curr], l0 * dt[:, None]
                )
            else:
                p = self._get_param_partial(
                    tlast, self.full_result[last], l1 * dt[:, None]
                )
            grad_result = self._accumulate(grad_result, p)

            # Only increment if we've hit an observation point
            if torch.any(tcurr <= times[j]):
                l1 = linear(tcurr, tlast, l1, l0, times[j])
                l0 = l1 + output_grad[j]
                j -= 1
            else:
                l0 = l1

        return grad_result

    def _step(self, t0, t1, dt, y0):
        """
        Actual "take a step" method

        Return the Jacobian even if it isn't used for the adjoint method

        Args:
          t0 (torch.tensor):     previous time
          t1 (torch.tensor):     next time
          dt (torch.tensor):     time increment
          y0 (torch.tensor):     previous state

        Returns:
          torch.tensor:   state at time :code:`t1`
        """
        raise NotImplementedError("_step not implemented in base class!")

    def _adjoint_update(self, dt, J, llast):
        """
        Update the *adjoint problem* to the next step

        The confusing thing here is that we must use the :math:`t_{n+1}`
        Jacobian as that's what we did on the forward pass, even though that's
        sort of backwards for the actual backward Euler method

        Args:
          dt (torch.tensor):    time step (defined positive!)
          J (torch.tensor):     cached Jacobian value --
                                actually :math:`\\frac{d\\dot{y}}{dy}` for this
                                method
          llast (torch.tensor): last value of the adjoint

        Returns:
          torch.tensor:           updated value of the adjoint at the "last" time step
        """
        raise NotImplementedError("_adjoint_update not implemented in base class!")

    def _accumulate(self, grad_results, partial):
        """
        Update the *parameter gradient* to the next step

        Args:
          grad_results (torch.tensor):    previous value of the gradient results
          partial (torch.tensor):         value of the parameter partials * dt

        Returns:
          tuple of torch.tensor:          updated parameter gradients at the "next" step
        """
        raise NotImplementedError("_accumulated not implemented in base class!")


class ExplicitSolver(FixedGridSolver):
    """
    Superclass of all explicit solvers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_curr = True  # Which location it needs the partial via autodiff

    def _adjoint_update(self, dt, J, llast):
        """
        Update the *adjoint problem* to the next step

        The confusing thing here is that we must use the :math:`t_{n+1}`
        Jacobian as that's what we did on the forward pass, even though that's
        sort of backwards for the actual backward Euler method

        Args:
          dt (torch.tensor):    time step (defined positive!)
          J (torch.tensor):     cached Jacobian value --
                                actually :math:`\\frac{d\\dot{y}}{dy}` for this
                                method
          llast (torch.tensor): last value of the adjoint

        Returns:
          torch.tensor:           updated value of the adjoint at the "last" time step
        """
        raise NotImplementedError("_adjoint_update not implemented in base class!")

    def _accumulate(self, grad_results, partial):
        """
        Update the *parameter gradient* to the next step

        Args:
          grad_results (torch.tensor):    previous value of the gradient results
          partial (torch.tensor):         value of the parameter partials * dt

        Returns:
          tuple of torch.tensor:          updated parameter gradients at the "next" step
        """
        raise NotImplementedError("_accumulated not implemented in base class!")


class ForwardEuler(ExplicitSolver):
    """
    Basic forward Euler integration
    """

    def _step(self, t0, t1, dt, y0):
        """
        Actual "take a step" method

        Return the Jacobian even if it isn't used for the adjoint method

        Args:
          t0 (torch.tensor):     previous time
          t1 (torch.tensor):     next time
          dt (torch.tensor):     time increment
          y0 (torch.tensor):     previous state

        Returns:
          torch.tensor:   state at time :code:`t1`
        """
        v, J = self.func(t0, y0)
        return y0 + v * dt[:, None], J

    def _adjoint_update(self, dt, J, llast):
        """
        Update the *adjoint problem* to the next step

        The confusing thing here is that we must use the :math:`t_{n+1}`
        Jacobian as that's what we did on the forward pass, even though that's
        sort of backwards for the actual backward Euler method

        Args:
          dt (torch.tensor):    time step (defined positive!)
          J (torch.tensor):     cached Jacobian value --
                                actually :math:`\\frac{d\\dot{y}}{dy}` for this
                                method
          llast (torch.tensor): last value of the adjoint

        Returns:
          torch.tensor:           updated value of the adjoint at the "last" time step
        """
        return (
            llast
            + torch.bmm(llast.view(self.batch_size, 1, self.prob_size), J)[:, 0, ...]
            * dt[..., None]
        )

    def _accumulate(self, grad_results, partial):
        """
        Update the *parameter gradient* to the next step

        Args:
          grad_results (torch.tensor):    previous value of the gradient results
          partial (torch.tensor):         value of the parameter partials * dt

        Returns:
          tuple of torch.tensor:          updated parameter gradients at the "next" step
        """
        return tuple(gi + pi for gi, pi in zip(grad_results, partial))


class ImplicitSolver(FixedGridSolver):
    """
    Superclass of all implicit solvers

    Keyword Args:
      iterative_linear_solver (bool):   if true, solve with an iterative linear scheme
      rtol (float):                     solver relative tolerance
      atol (float):                     solver absolute tolerance
      miter (int):                      maximum nonlinear iterations
      solver_method (string):           how to solve linear equations, `"diag"` or `"lu"`
                                        `"diag"` uses just the diagonal of the
                                        Jacobian -- the dumbest Newton-Krylov scheme ever.
                                        `"lu"` uses the full LU factorization, ignored
                                        if using an iterative linear solver
    """

    def __init__(
        self,
        *args,
        iterative_linear_solver=False,
        rtol=1.0e-6,
        atol=1.0e-10,
        miter=100,
        solver_method="lu",
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if not self.has_jac:
            raise ValueError(
                "Implicit solvers can only be used if the model returns the jacobian"
            )

        self.ils = iterative_linear_solver

        if not self.ils:
            self.solver_method = solver_method
        else:
            self.solver = solvers.PreconditionerReuseNonlinearSolver()

        self.rtol = rtol
        self.atol = atol
        self.miter = miter

        self.use_curr = False  # Which location it needs the partial via autodiff

    def _adjoint_update(self, dt, J, llast):
        """
        Update the *adjoint problem* to the next step

        The confusing thing here is that we must use the :math:`t_{n+1}`
        Jacobian as that's what we did on the forward pass, even though that's
        sort of backwards for the actual backward Euler method

        Args:
          dt (torch.tensor):    time step (defined positive!)
          J (torch.tensor):     cached Jacobian value --
                                actually :math:`\\frac{d\\dot{y}}{dy}` for this
                                method
          llast (torch.tensor): last value of the adjoint

        Returns:
          torch.tensor:           updated value of the adjoint at the "last" time step
        """
        raise NotImplementedError("_adjoint_update not implemented in base class!")

    def _accumulate(self, grad_results, partial):
        """
        Update the *parameter gradient* to the next step

        Args:
          grad_results (torch.tensor):    previous value of the gradient results
          partial (torch.tensor):         value of the parameter partials * dt

        Returns:
          tuple of torch.tensor:          updated parameter gradients at the "next" step
        """
        raise NotImplementedError("_accumulated not implemented in base class!")

    def _solve_system(self, system, guess):
        """
        Dispatch to solve the nonlinear system of equations

        Args:
          system (function):      function return R and J
          guess (torch.tensor):   initial guess at solution

        Returns:
          (torch.tensor, torch.tensor): the solution and Jacobian evaluated at the solution
        """
        if not self.ils:
            return solvers.newton_raphson(
                system,
                guess,
                linsolver=self.solver_method,
                rtol=self.rtol,
                atol=self.atol,
                miter=self.miter,
            )
        return self.solver.solve(
            system, guess, rtol=self.rtol, atol=self.atol, miter=self.miter
        )


class BackwardEuler(ImplicitSolver):
    """
    The classical backward Euler integration method
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _step(self, t0, t1, dt, y0):
        """
        Actual implementation of the time integration

        Return the final Jacobian, even though it's not used,
        for later use in the adjoint method

        Args:
          t0 (torch.tensor):     previous time
          t1 (torch.tensor):     next time
          dt (torch.tensor):     time increment
          y0 (torch.tensor):     previous state

        Returns:
          (torch.tensor, torch.tensor): next state and Jacobian + identity at that state
        """
        # Function which returns the residual and jacobian
        def RJ(x):
            f, df = self.func(t1, x)
            return x - y0 - f * dt[..., None], utility.add_id(-df * dt[..., None, None])

        return self._solve_system(RJ, y0.clone())

    def _adjoint_update(self, dt, J, llast):
        """
        Update the *adjoint problem* to the next step

        The confusing thing here is that we must use the :math:`t_{n+1}`
        Jacobian as that's what we did on the forward pass, even though
        that's sort of backwards for the actual implicit Euler method

        Args:
          dt (torch.tensor):      time step (defined positive!)
          Jlast (torch.tensor):   cached Jacobian value -- :math:`I - \\frac{d\\dot{y}}{dy}`
                                  for this method
          llast (torch.tensor):   last value of the adjoint

        Returns:
          torch.tensor:           updated adjoint problem
        """
        return torch.linalg.solve(J.transpose(1, 2), llast)

    def _accumulate(self, grad_results, partial):
        """
        Update the *parameter gradient* to the next step using the consistent
        integration method

        Args:
          grad_results (torch.tensor):    previous value of the gradient results
          partial (torch.tensor):         value of the parameter partials * dt

        Returns:
          tuple of torch.tensors:         updated parameter gradients
        """
        return tuple(gi + pi for gi, pi in zip(grad_results, partial))


# Available solver methods mapped to the objects
methods = {"forward-euler": ForwardEuler, "backward-euler": BackwardEuler, "block-backward-euler": BlockSolver}


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
    solver = methods[method](func, y0, **kwargs)

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
            ctx.save_for_backward(times)
            return y

    @staticmethod
    def backward(ctx, output_grad):
        """
        Args:
          ctx:            context object with state
          output_grad:    grads with which to dot product
        """
        with torch.no_grad():
            times = ctx.saved_tensors[0]
            grad_tuple = ctx.solver.rewind_adjoint(times, output_grad)
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
    if extra_params is None:
        extra_params = []
    adjoint_params = tuple(p for p in func.parameters()) + tuple(extra_params)

    solver = methods[method](func, y0, adjoint_params=adjoint_params, **kwargs)

    wrapper = IntegrateWithAdjoint()

    return wrapper.apply(solver, times, *adjoint_params)
