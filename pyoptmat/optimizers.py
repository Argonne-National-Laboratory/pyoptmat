import torch
from torch import optim

import math

class Adam2BFGS(optim.Optimizer):
    """Use Adam for awhile to build an approximate Hessian for BFGS

    Args:
        params (iterable): an iterable collection of :class:`torch.Tensor`
            or :class:`dict`.  Which tensors to optimize.
        nwarmup (int): number of warmup steps to take with Adam before 
            switching to L-BFGS
        adam_params (dict), optional: parameters for Adam
        lbfgs_params (dict), optional: parameters for L-BFGS.  Note that 
        lbfgs_params['history_size'], if provided, should be 
            greater than nwwarmup
    """
    def __init__(self, params, nwarmup, adam_params = dict(),
            lbfgs_params = dict()):
        # Or else it consumes the parameters
        params = list(params)
        super().__init__(params, dict())

        if 'history_size' in lbfgs_params and lbfgs_params['history_size'] < nwarmup:
            raise ValueError("Provided history_size = %i is less than the number of "
                    "Adam warmup iterations %i.  This doesn't make sense." % 
                    (lbfgs_params['history_size'], nwarmup))
        elif 100 < nwarmup:
            # 100 is the default value
            raise ValueError("Default history size = 100 greater than the provided "
                    "number of Adam warmup iterations %i.  This doesn't make sense." %
                    nwarmup)

        self.nwarmup = nwarmup
        self.adam = optim.Adam(params, **adam_params)
        self.lbfgs = optim.LBFGS(params, **lbfgs_params)

        # This all should be in state...
        self.old_dirs = []
        self.old_stps = []
        self.ro = []
        self.H_diag = 1.0
        self.al = [None] * self.nwarmup
        self.i = 0
   
        # Only allow one group
        assert(len(self.param_groups) == 1)
        # Don't allow custom line search
        assert(self.lbfgs.param_groups[0]['line_search_fn'] is None)

        # Get the learning rate to use during the hessian update
        # Should this be the L-BFGS learning rate or the Adam learning rate?
        self.lr_lbfgs = self.lbfgs.param_groups[0]['lr']

    @property
    def _lbfgs_state(self):
        """The LBFGS solver caches the state as the first parameter
        """
        return self.lbfgs.state[self.lbfgs._params[0]]

    @torch.no_grad()
    def step(self, closure, **kwargs):
        """Run the optimization

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the less.

        kwargs get passed to Adam
        """
        if self.i < self.nwarmup:
            print("Warmup step")
            self.lv = self._warmup_step(closure, **kwargs)
        else:
            if self.i == self.nwarmup:
                self._prime_lbfgs()
            print("LBFGS step")
            self.lv = self.lbfgs.step(closure)

        self.i += 1

        return self.lv

    def _prime_lbfgs(self):
        """Prime LBFGS with the values we stored
        """
        print("Priming LBFGS")
        self._lbfgs_state['d'] = self.d
        self._lbfgs_state['t'] = self.t
        self._lbfgs_state['old_dirs'] = self.old_dirs
        self._lbfgs_state['old_stps'] = self.old_stps
        self._lbfgs_state['ro'] = self.ro
        self._lbfgs_state['H_diag'] = self.H_diag
        self._lbfgs_state['prev_flat_grad'] = self.prev_flat_grad
        self._lbfgs_state['prev_loss'] = self.lv
        self._lbfgs_state['n_iter'] = self.i

    def _warmup_step(self, closure, **kwargs):
        """An Adam step with  L-BFGS hessian update
        """
        # Take Adam step
        lv = self.adam.step(closure = closure, **kwargs)

        # Get the flat gradient
        self.flat_grad = self.lbfgs._gather_flat_grad()

        # If the first warmup iteration just setup previous grad
        if self.i == 0:
            self.prev_flat_grad = self.flat_grad.clone(memory_format=torch.contiguous_format)
            self.d = self.flat_grad.neg()
            self.prev_loss = lv
            self.t = min(1., 1. / self.flat_grad.abs().sum()) * self.lr_lbfgs
            return lv

        # Do the Hessian calculation
        y = self.flat_grad.sub(self.prev_flat_grad)
        s = self.d.mul(self.t)
        ys = y.dot(s)
        self.H_diag = ys / y.dot(y)
        self.old_dirs.append(y)
        self.old_stps.append(s)
        self.ro.append(1.0 / ys)

        q = self.flat_grad.neg()
        num_old = len(self.old_dirs)
        for j in range(num_old -1, -1, -1):
            self.al[j] = self.old_stps[j].dot(q) * self.ro[j]
            q.add_(self.old_dirs[j], alpha = -self.al[j])
        self.d = torch.mul(q, self.H_diag)

        self.t = self.lr_lbfgs

        # Copy flat grad
        self.prev_flat_grad.copy_(self.flat_grad)

        return lv
