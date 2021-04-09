"""."""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from ..utils.tensor import make_symmetric_and_traceless, make_traceless


class IsingVectorModel(nn.Module):
    """One layer, one update step. Inputs/outputs of this thing should be fixed-pointed.

    

    This module is a container for the parameters defining the system:
      - (trainable) pairwise coupling matrix J between spins
      - (trainable) Gaussian prior means and variances for the spins
    Given these parameters (couplings and priors), the adaptive TAP framework
    provides a closed-form solution for the Gibbs free energy and sets of
    equations that should be solved self-consistently for a fixed point.
    The algorithm is equivalent to expectation propagation (see Section 4.3 in
    https://arxiv.org/abs/1409.6179) and boils down to matching the first and
    second moments.

    The variables solved for during fixed-point are:
      - spin_mean (B, N, dim)
      - spin_var (B, N, 1)
      x cavity_mean (B, N, dim)
      - cavity_var (B, N, 1)
    The fixed-point average magnetizations `spin_mean` are interpreted as the
    updated "queries" in transformer parlance. These updated queries have not
    been just aligned to the "nearest sources" but are the result of interal
    dynamics of the spin system (within the approximation of mean-field with
    linear corrections).
    """

    def __init__(
        self,
        num_spins,  # number of spins (~ max_seq_len)
        dim,  # local dimension of spins
        J_init_mean=0.0,
        J_init_std=1.0,
        J_symmetric=True,  # undirected or directed interaction graph
        J_trainable=False,
        prior_init_std=1.0,  # can be float or list of num_spins floats
        prior_trainable=False,
    ):
        super().__init__()

        self._init_J(
            num_spins,
            init_std=J_init_std,
            symmetric=J_symmetric,
            trainable=J_trainable,
        )

        self.prior_init_std = prior_init_std
        self.J_symmetric = J_symmetric

    def _init_J(self, num_spins, init_std, symmetric, trainable):
        """Initialize random coupling matrix."""
        J = init_std * torch.randn(num_spins, num_spins)  # dtype=torch.float64)
        if trainable:
            self._J = nn.Parameter(J)
        else:
            self.register_buffer("_J", J)

    @property
    def J(self):
        if self.J_symmetric:
            return make_symmetric_and_traceless(self._J)
        return make_traceless(self._J)

    @property
    def gibbs_free_energy(self):
        pass

    def susceptibility(cavity_vars,):
        """Calculate covariances for spins."""

    def _check_psd_susceptibility(self):
        pass  # calculate susceptibility and check if all eigvals are positive

    def get_initial_guess(self, x):
        return [
            torch.zeros_like(x),  # spin_mean
            torch.ones(
                (x.size(0), x.size(1), 1), device=x.device, dtype=x.dtype
            ),  # spin_var
            torch.zeros(
                (x.size(0), x.size(1), 1), device=x.device, dtype=x.dtype
            ),  # cavity_var
        ]

    def forward(self, state, source, *args):
        """Gets a list, spits out a list."""
        spin_mean, spin_var, cavity_var = state
        bsz = spin_mean.size(0)

        cavity_mean = (
            torch.einsum("n m, b m d -> b n d", self.J, spin_mean)
            - cavity_var * spin_mean
        )
        # print("next_cavity_mean", next_cavity_mean)
        print(self.J)

        prefactor = 1.0 / (self.prior_init_std ** 2 - cavity_var)  # .unsqueeze(-1)
        # print(prefactor.shape, next_cavity_mean.shape, source.shape)
        next_spin_mean = prefactor * (cavity_mean + source)
        next_spin_var = prefactor

        bla = (cavity_var + 1.0 / next_spin_var).squeeze(-1)
        lambda_diag = torch.diag_embed(bla)
        X = lambda_diag - self.J.unsqueeze(0).repeat(bsz, 1, 1)
        # X = 0.5 * (X + X.permute(0, 2, 1))
        # print(X)
        # print(torch.symeig(X[0]))

        # L = torch.cholesky(X)
        ones = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
        # print(L, ones)
        # out = torch.cholesky_solve(ones, L)
        out, _ = torch.solve(ones, X)
        # print(out)
        # Take batch diagonal
        # print(out.shape)
        out = torch.diagonal(out, dim1=-2, dim2=-1)
        # print(out.shape, bla.shape)
        next_cavity_var = bla / out

        # print("next_cavity_var", next_cavity_var)
        print("next_spin_mean", next_spin_mean)
        # print("next_spin_var", next_spin_var)
        # print(self.J)

        return [
            next_spin_mean,
            next_spin_var,
            next_cavity_var,
        ]
