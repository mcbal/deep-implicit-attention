import torch
import torch.nn as nn

from .deq import _DEQModule
from .solvers import anderson
from .utils import make_symmetric_and_traceless, make_traceless


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class IsingGaussianAdaTAP(_DEQModule):
    """Ising-like vector model with Gaussian prior over spins.

    This module is a container for the parameters defining the system:
      - pairwise coupling matrix weights between spins
      - Gaussian prior for the spins

    Given these parameters (couplings and priors), the adaptive TAP framework
    provides a closed-form solution for the Gibbs free energy and sets of
    equations that should be solved self-consistently for a fixed point.
    The algorithm is equivalent to expectation propagation (see Section 4.3 in
    https://arxiv.org/abs/1409.6179) and boils down to matching the first and
    second moments assuming a Gaussian cavity distribution.

    To use this module, wrap it in `modules.DEQFixedPoint`.
    """

    def __init__(
        self,
        num_spins,
        dim,
        weights_init_std=1.0,
        weights_symmetric=True,
        weights_training=True,
        prior_init_std=1.0,
        prior_training=False,
        solver=anderson,
        solver_tol=1e-4,
        solver_max_iter=30,
        lin_response_correction=True,
    ):
        super().__init__()

        self._init_weights(
            num_spins, init_std=weights_init_std, training=weights_training,
        )
        self.weights_symmetric = weights_symmetric
        self.prior_init_std = prior_init_std

        self.solver = solver
        self.solver_tol = solver_tol
        self.solver_max_iter = solver_max_iter

        self.lin_response_correction = lin_response_correction

    def _init_weights(self, num_spins, init_std, training):
        """Initialize random coupling matrix."""
        weights = init_std * torch.randn(num_spins, num_spins)
        if training:
            self._weights = nn.Parameter(weights)
        else:
            self.register_buffer("_weights", weights)

    @property
    def weights(self):
        if self.weights_symmetric:
            return make_symmetric_and_traceless(self._weights)
        return make_traceless(self._weights)

    def gibbs_free_energy(self, z):
        raise NotImplementedError()

    def get_initial_guess(self, x):
        return [
            torch.zeros_like(x),  # spin_mean
            torch.ones(
                (x.size(0), x.size(1), 1), device=x.device, dtype=x.dtype
            ),  # spin_var
        ]

    def _solve_cavity(self, spin_mean, spin_var):
        if self.lin_response_correction:
            bsz = spin_mean.size(0)

            def v_fun(v):
                big_lambda = (v.unsqueeze(-1) + 1.0 / spin_var).squeeze(-1)
                X = torch.diag_embed(big_lambda) - self.weights[None, :, :].repeat(
                    bsz, 1, 1
                )
                ones = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)[
                    None, :, :
                ].repeat(bsz, 1, 1)
                out, _ = torch.solve(ones, X)
                # Check for negative eigenvalues (instability of mean-field solution).
                # print(torch.eig(out[0]), torch.dist(ones, X.matmul(out)))
                new_v = big_lambda - 1.0 / torch.diagonal(out, dim1=-2, dim2=-1)
                return new_v

            result = self.solver(
                v_fun,
                torch.zeros_like(spin_var).squeeze(-1),
                max_iter=self.solver_max_iter,
                tol=self.solver_tol,
            )
            cav_var = result["result"].unsqueeze(-1)
        else:
            cav_var = torch.zeros_like(spin_var)

        cav_mean = (
            torch.einsum("n m, b m d -> b n d", self.weights, spin_mean)
            - cav_var * spin_mean
        )
        return cav_mean, cav_var  # (bsz, num_spins, dim), (bsz, num_spins, 1)

    def forward(self, z, x, *args):
        spin_mean, spin_var = self.unpack_state(z)

        cav_mean, cav_var = self._solve_cavity(spin_mean, spin_var)

        pf = self.prior_init_std ** 2 / (1 - self.prior_init_std ** 2 * cav_var)
        next_spin_mean, next_spin_var = pf * (cav_mean + x), pf

        return self.pack_state([next_spin_mean, next_spin_var])
