import copy
import torch.nn as nn

from .adatap import IsingVectorModel
from .deq import DEQWrapper
from ..solvers import AndersonSolverOptions


class FixedPointAttention(nn.Module):
    """Attention module based on an Ising-like model of `num_spins` classical
    vector spins of local dimension `dim` (without unit-length constraints).

    Couplings are randomly initialized ~ N(0, `J_init_std` / sqrt(num_spins))
    """

    def __init__(
        self,
        *,
        num_spins,  # number of spins (~ max_seq_len)
        dim,  # local dimension of spins
        J_init_mean=0.0,
        J_init_std=1.0,
        J_symmetric=True,  # undirected or directed interaction graph
        J_trainable=False,
        prior_init_std=1.0,  # can be float or list of num_spins floats
        prior_trainable=False,
        solver_max_iter=100,
        solver_tol=1e-6,
    ):
        super().__init__()

        # Initialize system
        self.model = IsingVectorModel(
            num_spins,
            dim,
            J_init_mean=J_init_mean,
            J_init_std=J_init_std,
            J_symmetric=J_symmetric,  # undirected or directed interaction graph
            J_trainable=J_trainable,
            prior_init_std=prior_init_std,  # can be float or list of num_spins floats
            prior_trainable=prior_trainable,
        )

        self.model_copy = copy.deepcopy(self.model)
        for param in self.model_copy.parameters():
            param.requires_grad_(False)
        self.deq = DEQWrapper(self.model, self.model_copy)

        # Store solver options.
        self.solver_opts = AndersonSolverOptions(
            tol=solver_tol, max_iter=solver_max_iter,
        )

    def _fixed_point(self, z0, x, **kwargs):
        """Solve system for fixed point.

        :param x: input source injections at every site (B, N, dim)
        :param kwargs: all other things
        :return: z_star: equilibrium state (list of fixed point variables)
        """
        z_star = self.deq(z0, x, **kwargs)
        return z_star

    def forward(self, x, **kwargs):
        """bla
        :return: equilibrium state (list of fixed point variables)
        """
        z0 = kwargs.get("z0", self.model.get_initial_guess(x))
        updated_kwargs = {**self.solver_opts.to_dict(), **kwargs}
        updated_kwargs = {**kwargs, **updated_kwargs}
        out = self._fixed_point(z0, x, **updated_kwargs)
        return out
