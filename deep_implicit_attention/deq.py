from abc import ABCMeta, abstractmethod
from functools import reduce

import torch
import torch.nn as nn
import torch.autograd as autograd

from .utils import filter_kwargs


class _DEQModule(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.shapes = None

    def pack_state(self, z_list):
        """
        Transform list of batched tensors into batch of vectors.
        """
        self.shapes = [t.shape[1:] for t in z_list]
        bsz = z_list[0].shape[0]
        z = torch.cat([t.reshape(bsz, -1) for t in z_list], dim=1)
        return z

    def unpack_state(self, z):
        """
        Transform batch of vectors into list of batched tensors.
        """
        assert self.shapes is not None
        bsz, z_list = z.shape[0], []
        start_idx, end_idx = 0, reduce(lambda x, y: x * y, self.shapes[0])
        for i in range(len(self.shapes)):
            z_list.append(z[:, start_idx:end_idx].view(bsz, *self.shapes[i]))
            if i < len(self.shapes) - 1:
                start_idx = end_idx
                end_idx += reduce(lambda x, y: x * y, self.shapes[i + 1])
        return z_list

    @abstractmethod
    def _initial_guess(self, x):
        """Return an initial guess for the fixed-point state based on shape of `x`."""
        pass

    @abstractmethod
    def forward(self, z, x, *args):
        """Implement f(z_{n}, x) -> z_{n+1}."""
        pass


class DEQFixedPoint(nn.Module):
    _default_kwargs = {
        'solver_fwd_max_iter': 30,
        'solver_fwd_tol': 1e-4,
        'solver_bwd_max_iter': 30,
        'solver_bwd_tol': 1e-4,
    }

    def __init__(self, fun, solver, output_elements=[0], **kwargs):
        super().__init__()
        self.fun = fun
        self.solver = solver
        self.output_elements = output_elements
        self.kwargs = self._default_kwargs
        self.kwargs.update(**kwargs)

    def _fixed_point(self, z0, x, *args, **kwargs):
        """Find fixed-point of `fun` given `z0` and `x`."""

        # Compute forward pass: find equilibrium state
        with torch.no_grad():
            out = self.solver(
                lambda z: self.fun(z, x, *args),
                z0,
                **filter_kwargs(kwargs, 'solver_fwd_'),
            )
            z = out['result']
            if kwargs.get('debug', False):
                print(f"{out['rel_trace'][0]} -> {out['rel_trace'][-1]}")
                from .utils import log_plot
                log_plot(out['rel_trace'])

        if self.training:
            # Re-engage autograd tape at equilibrium state
            z = self.fun(z, x, *args)
            # Set up Jacobian vector product (without additional forward calls)
            z_bwd = z.clone().detach().requires_grad_()
            fun_bwd = self.fun(z_bwd, x, *args)

            def backward_hook(grad):
                out = self.solver(
                    lambda y: autograd.grad(
                        fun_bwd, z_bwd, y, retain_graph=True)[0]
                    + grad,
                    torch.zeros_like(grad),
                    **filter_kwargs(kwargs, 'solver_bwd_'),
                )
                g = out['result']
                # [DEBUG] insert statements here for backward pass inspection
                return g

            z.register_hook(backward_hook)

        return z

    def forward(self, x, *args, **kwargs):
        # Merge default kwargs with incoming runtime kwargs.
        kwargs = {**self.kwargs, **kwargs}
        # Get list of initial guess tensors and reshape into a batch of vectors
        z0 = self.fun.pack_state(self.fun._initial_guess(x))
        # Find equilibrium vectors
        z_star = self._fixed_point(z0, x, *args, **kwargs)
        # Return (subset of) list of tensors of original input shapes
        out = [self.fun.unpack_state(z_star)[i] for i in self.output_elements]
        return out[0] if len(out) == 1 else out
