"""
  Module defining the key objects and functions to integrate ODEs
  and provide the sensitivity of the results with respect to the
  model parameters either through backpropogation AD or the adjoint method.
"""

import torch
from tqdm import tqdm

import warnings

from pyoptmat.utility import timeseries_interpolate_batch_times

def linear(t0, t1, y0, y1, t):
  """
    Helper function for linear interpolation between two points

    Args:
      t0:     first "x" point
      t1:     second "x" point
      y0:     first "y" point
      y1:     second "y" point
      t:      target "x" point
  """
  n = y0.dim() - 1
  return y0 + (y1-y0) / (t1-t0)[(...,)+(None,)*n] * (t - t0)[(...,)+(None,)*n]

class FixedGridSolver:
  """
    Superclass of all solvers that use a fixed grid (versus an adaptive method)

    Args:
      func:                 function returning the time rate of change and,
                            optionally, the jacobian
      y0:                   initial condition
      substep (optional):   subdivide each provided timestep into some number
                            of subdivisions for integration
      jit_mode (optional):  if true do various dangerous things to fix the model structure
  """   
  def __init__(self, func, y0, adjoint_params = None, **kwargs):
    # Store basic info about the system
    self.func = func
    self.y0 = y0

    self.batch_size = self.y0.shape[0]
    self.prob_size = self.y0.shape[1]

    self.substep = kwargs.pop('substep', None)
    self.jit_mode = kwargs.pop('jit_mode', False)

    # Store for later
    self.adjoint_params = adjoint_params
    
    # Sort out if the function is providing the jacobian
    self.has_jac = True

    if not self.jit_mode:
      fake_time = torch.zeros(self.batch_size, device = y0.device)
      try:
        a, b = self.func(fake_time, self.y0)
      except ValueError:
        self.has_jac = False
      if not self.has_jac:
        a = self.func(fake_time, self.y0)

      if a.dim() != 2:
        raise ValueError("ODE solvers require batched functions returning (nbatch, nvars)!")

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
        t:      time you want
        y:      state you want
        og:     thing to dot with
    """
    with torch.enable_grad():
      ydot = self.func(t, y)[0]
      # Retain graph seems to be needed for solvers like L-BFGS
      # which go twice through the function
      return torch.autograd.grad(ydot, self.adjoint_params,
          og, retain_graph = True)

  def _construct_grid(self, t):
    """
      Construct the grid for substepped problems

      Args:
        t:      initial timesteps
    """
    if self.substep and self.substep > 1:
      nshape = list(t.shape)
      nshape[0] = (nshape[0]-1) * self.substep

      grid = torch.empty(nshape, device = t.device)
      incs = torch.linspace(0,1,self.substep+1, device = t.device)[:-1]

      i = 0
      for t1,t2 in zip(t[:-1],t[1:]):
        grid[i:i+self.substep] = (incs * (t2-t1).unsqueeze(-1) + t1.unsqueeze(-1)).T
        i += self.substep

      grid[-1] = t[-1]

      return grid

    return t

  def integrate(self, t, cache_adjoint = False):
    """
      Main method: actually integrate through the provided times

      Args:
        t:                          timesteps to report results at
        cache_adjoint (optional):   store the info we'll need for the
                                    adjoint pass
    """
    # Basic error checking
    if not self.jit_mode:
      if t.dim() != 2 or t.shape[1] != self.batch_size:
        raise ValueError("Expected times to be a ntime x batch_size array!")
    
    # Construct the substepped grid
    times = self._construct_grid(t)
    
    # Setup "real" results
    result = torch.empty(t.shape[0], *self.y0.shape, dtype = self.y0.dtype,
        device = self.y0.device)
    result[0] = self.y0
    
    # Setup "cached" results, if required
    if cache_adjoint:
      self.full_result = torch.empty(times.shape[0], *self.y0.shape, dtype = self.y0.dtype,
          device = self.y0.device)
      self.full_result[0] = self.y0
      self.full_jacobian = torch.empty(times.shape[0], self.batch_size, self.prob_size, 
          self.prob_size, dtype = self.y0.dtype, device = self.y0.device)
      self.full_times = times
    
    # Start integrating!
    y0 = self.y0
    j = 1
    for k,(t0, t1) in enumerate(zip(times[:-1], times[1:])):
      y1, J = self._step(t0, t1, t1-t0, y0)
      
      # Save the full set of results and the Jacobian, if required for backward pass
      if cache_adjoint:
        self.full_result[k+1] = y1
        self.full_jacobian[k+1] = J
      
      # Interpolate to results point, if requested
      if torch.any(t1 >= t[j]): # This could be slow
        result[j] = linear(t0, t1, y0, y1, t[j])
        j += 1 

      y0 = y1
    
    result[-1] = y1 # There may be a more elegant way of doing this...
    
    return result

  def rewind_adjoint(self, times, output_grad):
    """
      Rewind the solve the get the sensitivities via the adjoint method

      Args:
        times:          actual times required by the solver
        output_grad:    dot product tensor
    """
    # Start from *end* of the output_grad -- going backwards
    l0 = output_grad[-1]
    # Useful for indexing
    nt = self.full_times.shape[0]
    # Gradient results
    grad_result = tuple(torch.zeros(p.shape, 
      device = times.device) for p in self.adjoint_params)

    # Target output time
    j = times.shape[0]-2

    # Run *backwards* through time
    for curr in range(nt-2,-1,-1):
      # Setup all the state so I don't get confused
      last = curr + 1 # Very confusing lol
      tcurr = self.full_times[curr]
      tlast = self.full_times[last]
      dt = tlast - tcurr  # I define this to be positive
      # Take the adjoint step
      l1 = self._adjoint_update(dt, self.full_jacobian[last], l0)

      # Accumulate the gradients
      # The l0, l1 thing confuses me immensely, but is in fact correct
      # Note where the dt goes, this is to avoid branching later when summing
      if self.use_curr:
        p = self._get_param_partial(tcurr, self.full_result[curr], l0 * dt[:,None])
      else:
        p = self._get_param_partial(tlast, self.full_result[last], l1 * dt[:,None])
      grad_result = self._accumulate(grad_result, p)

      # Only increment if we've hit an observation point
      if torch.any(tcurr <= times[j]):
        l1 = linear(tcurr, tlast, l1, l0, times[j])
        l0 = l1 + output_grad[j]
        j -= 1
      else:
        l0 = l1

    return grad_result

class ExplicitSolver(FixedGridSolver):
  """
    Superclass of all explicit solvers
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.use_curr = True # Which location it needs the partial via autodiff

class ForwardEuler(ExplicitSolver):
  """
    Basic forward Euler integration
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def _step(self, t0, t1, dt, y0):
    """
      Actual "take a step" method

      Return the Jacobian even if it isn't used for the adjoint method

      Args:
        t0:     previous time
        t1:     next time
        dt:     time increment
        y0:     previous state
    """
    v, J = self.func(t0, y0)
    return y0 + v * dt[:,None], J

  def _adjoint_update(self, dt, J, llast):
    """
      Update the *adjoint problem* to the next step

      The confusing thing here is that we must use the :math:`t_{n+1}`
      Jacobian as that's what we did on the forward pass, even though that's
      sort of backwards for the actual implicit Euler method

      Args:
        dt:     time step (defined positive!)
        J:      cached Jacobian value -- actually :math:`\\frac{d\\dot{y}}{dy}` for this method
        llast:  last value of the adjoint
    """
    return llast + torch.bmm(llast.view(self.batch_size, 1, self.prob_size), J)[:,0,...] * dt[...,None]

  def _accumulate(self, grad_results, partial):
    """
      Update the *parameter gradient* to the next step

      Args:
        grad_results:   previous value of the gradient results
        partial:        value of the parameter partials * dt
    """
    return tuple(gi + pi for gi, pi in zip(grad_results, partial))
    
class ImplicitSolver(FixedGridSolver):
  """
    Superclass of all implicit solvers

    Args:
      rtol (optional):              solver relative tolerance
      atol (optional):              solver absolute tolerance
      miter (optional):             maximum nonlinear iterations
      solver_method (optional):     how to solve linear equations, `"diag"` or
                                    `"lu"`
                                    `"diag"` uses just the diagonal of the
                                    Jacobian -- the dumbest NK scheme ever.
                                    `"lu"` uses the full LU factorization
  """
  def __init__(self, *args, rtol = 1.0e-6, atol = 1.0e-10, miter = 100,
      solver_method = "lu",  **kwargs):
    super().__init__(*args, **kwargs)

    if not self.has_jac:
      raise ValueError("Implicit solvers can only be used if the model returns the jacobian")
    
    if solver_method not in ["lu", "diag"]:
      raise ValueError("Solver method %s not in available options!" % solver_method)
    self.solver_method = solver_method

    self.rtol = rtol
    self.atol = atol
    self.miter = miter

    self.use_curr = False # Which location it needs the partial via autodiff

  def _solve_system(self, system, guess):
    """
      Dispatch to solve the nonlinear system of equations

      Args:
        system:     function return R and J
        guess:      initial guess at solution
    """
    if self.solver_method == "diag" or self.solver_method == "lu":
      return self._solve_system_nr(system, guess)
    else:
      raise NotImplementedError("Unknown solver method!")

  def _solve_linear_system(self, A, b):
    """
      Dispatch to solve a linear system of equations

      Args:
        A:      batched matrices
        b:      batched right hand sides
    """
    if self.solver_method == "diag":
      return b / torch.diagonal(A, dim1=-2, dim2=-1)
    elif self.solver_method == "lu":
      return torch.linalg.solve(A, b)
    else:
      raise ValueError("Unknown solver method!")

  def _add_id(self, df):
    """
      Add the identity to a tensor with the shape of the jacobian

      Args:
        df:     batched `(n,m,m)` tensor
    """
    return df + torch.eye(df.shape[1], device = df.device).reshape((1,)  + df.shape[1:]).repeat(df.shape[0],1,1)

  def _solve_system_nr(self, system, x0):
    """
      Solve a system of nonlinear equations using Newton's method

      Args:
        system:     function that returns R,J = F(x)
        x0:         initial guess at solution
    """
    x = x0
    R, J = system(x)

    nR = torch.norm(R, dim = -1)
    nR0 = nR
    i = 0

    while (i < self.miter) and torch.any(nR > self.atol) and torch.any(nR / nR0 > self.rtol):
      x -= self._solve_linear_system(J, R)
      R, J = system(x)
      nR = torch.norm(R, dim = -1)
      i += 1

    if i == self.miter:
      warnings.warn("Implicit solve did not succeed.  Results may be inaccurate...")

    return x, J

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
        t0:     previous time
        t1:     next time
        dt:     time increment
        y0:     previous state
    """
    # Function which returns the residual and jacobian 
    def RJ(x):
      f, df = self.func(t1, x)
      return x - y0 - f * dt[...,None], self._add_id(-df * dt[...,None,None])

    return self._solve_system(RJ, y0.clone())

  def _adjoint_update(self, dt, J, llast):
    """
      Update the *adjoint problem* to the next step

      The confusing thing here is that we must use the :math:`t_{n+1}`
      Jacobian as that's what we did on the forward pass, even though 
      that's sort of backwards for the actual implicit Euler method

      Args:
        dt:     time step (defined positive!)
        Jlast:  cached Jacobian value -- :math:`I - \\frac{d\\dot{y}}{dy}`
                for this method
        llast:  last value of the adjoint
    """
    return torch.linalg.solve(J.transpose(1,2), llast)
  
  def _accumulate(self, grad_results, partial):
    """
      Update the *parameter gradient* to the next step using the consistent
      integration method

      Args:
        grad_results:   previous value of the gradient results
        partial:        value of the parameter partials * dt
    """
    return tuple(gi + pi for gi, pi in zip(grad_results, partial)) 

# Available solver methods mapped to the objects
methods = {"forward-euler": ForwardEuler, "backward-euler": BackwardEuler}

def odeint(func, y0, times, method = 'backward-euler', extra_params = None,
    **kwargs):
  """
    Solve the ordinary differential equation defined by the function func
    
    Input variables and initial condition have shape `(nbatch, nvar)`

    Input times have shape `(ntime, nbatch)`

    Output history has shape `(ntime, nbatch, nvar)`

    Args:
      func:             returns tensor defining the derivative y_dot and,
                        optionally, the jacobian as a second return value 
      y0:               initial conditions
      times:            time locations to solve for
      method:           integration method, currently `"backward-euler"` or 
                        `"forward-euler"`
      extra-params:     not used here, just maintains compatibility with the
                        adjoint interface
      kwargs:           keyword arguments passed on to specific solver methods
  """
  solver = methods[method](func, y0, **kwargs)
  
  return solver.integrate(times)

class IntegrateWithAdjoint(torch.autograd.Function):
  """
    Wrapper to convince the thing that it needs to use the hard-coded sensitivity
  """
  @staticmethod
  def forward(ctx, solver, times, *params):
    """
      Args:
        ctx:        context object we can use to stash state
        solver:     ODE Solver object to use
        times:      times to hit
    """
    with torch.no_grad():
      # Do our first pass and get full results
      y = solver.integrate(times, cache_adjoint = True)
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

def odeint_adjoint(func, y0, times, method = 'backward-euler',
    extra_params = [], **kwargs):
  """
    Solve the ordinary differential equation defined by the function func

    Calculate the gradient with the adjoint trick
    
    Input variables and initial condition have shape `(nbatch, nvar)`

    Input times have shape `(ntime, nbatch)`

    Output history has shape `(ntime, nbatch, nvar)`

    Args:
      func:         returns tensor defining the derivative :math:`\\dot{y}` and,
                    optionally, the jacobian as a second return value 
      y0:           initial conditions
      times:        time locations to solve for
      method:       integration method, currently `"backward-euler"` or 
                    `"forward-euler"`
      extra_params: any additional pytorch parameters that need to be included
                    in the adjoint calculation that are not determinable
                    via introspection
      kwargs:   keyword arguments passed on to specific solver methods
  """
  adjoint_params = tuple(p for p in func.parameters()) + tuple(extra_params)
  
  solver = methods[method](func, y0, adjoint_params = adjoint_params, **kwargs)

  wrapper = IntegrateWithAdjoint()

  return wrapper.apply(solver, times, *adjoint_params)
