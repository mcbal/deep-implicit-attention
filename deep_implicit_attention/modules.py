import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

from .deq import _DEQModule
from .utils import batched_eye_like


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


class GeneralizedIsingGaussianAdaTAP(_DEQModule):
    """Ising-like vector model with Gaussian prior over spins.

    This module is a container for the parameters defining the system:
      - pairwise coupling matrix weight between spins
      - Gaussian prior for the spins

    Given these parameters (couplings and priors), the adaptive TAP framework
    provides a closed-form solution for the Gibbs free energy and sets of
    equations that should be solved self-consistently for a fixed point.
    The algorithm is equivalent to expectation propagation (see Section 4.3 in
    https://arxiv.org/abs/1409.6179) and boils down to matching the first and
    second moments assuming a Gaussian cavity distribution.

    To use this module, wrap it in `modules.DEQFixedPoint`.

    TODO:
      - Make single-
    """

    def __init__(
        self,
        num_spins,  # number of vector spin degrees of freedom
        dim,  # vector dimension of degrees of freedom
        weight_init_std=None,  # std of random Gaussian weight initialization
        weight_symmetric=True,  # enforce symmetric weight
        weight_training=True,  # turn weight into parameter
        lin_response=True,  # toggle linear response correction to mean-field
    ):
        super().__init__()

        self._init_weight(
            num_spins,
            dim,
            init_std=(
                weight_init_std
                if weight_init_std is not None
                else 1.0 / np.sqrt(num_spins * dim ** 2)
            ),
            training=weight_training,
        )
        self.weight_symmetric = weight_symmetric
        self.prior_inv_cov = batched_eye_like(torch.zeros(num_spins, dim, dim))
        self.lin_response = lin_response

    def _init_weight(self, num_spins, dim, init_std, training):
        """Initialize random coupling matrix."""
        weight = init_std * torch.randn(num_spins, num_spins, dim, dim)
        if training:
            self._weight = nn.Parameter(weight)
        else:
            self.register_buffer("_weight", weight)

    # @lru_cache(maxsize=1)
    def weight(self):
        # https://pytorch.org/tutorials/intermediate/parametrizations.html
        num_spins, dim = self._weight.size(0), self._weight.size(2)
        weight = 0.5 * (self._weight + self._weight.permute([0, 1, 3, 2]))
        weight = 0.5 * (weight + weight.permute([1, 0, 2, 3]))
        mask = batched_eye_like(torch.zeros(dim ** 2, num_spins, num_spins))
        mask = mask.permute([1, 2, 0]).reshape(num_spins, num_spins, dim, dim)
        weight = (1.0 - mask) * weight
        return weight

    def _initial_guess(self, x):
        """
        Return initial guess tensors.
        """
        return [
            torch.zeros_like(x),
            torch.zeros((*x.shape, x.shape[-1]), device=x.device, dtype=x.dtype),
        ]

    def _spin_mean_var(self, x, cav_mean, cav_inv_cov):
        """
        
        These expressions are obtained from integrating the single-site partition function,
        where a Gaussian prior has been . Inserting a X prior for scalar degrees of freedom
        would give a cosh(...)-expression for the single-site partition function and hence
        a spin expectation value involving tanh(...) (see e.g.)."""
        # prefactor = 1.0 / (1.0 / (self.prior_init_std ** 2) - cav_inv_cov)
        X = self.prior_inv_cov - cav_inv_cov  # (N, d, d)
        ones = batched_eye_like(X)
        pf = torch.solve(ones, X).solution
        # print(prefactor)
        # breakpoint()
        spin_mean = torch.einsum("n d e, b n d -> b n e", pf, (cav_mean + x))
        spin_var = pf
        return spin_mean, spin_var

    def forward(self, z, x, *args):
        spin_mean, cav_inv_cov = self.unpack_state(z)

        cav_mean = torch.einsum(
            "n m d e, b m d -> b n e", self.weight(), spin_mean
        ) - torch.einsum("b n d e, b n d -> b n e", cav_inv_cov, spin_mean)

        spin_mean, spin_var = self._spin_mean_var(x, cav_mean, cav_inv_cov[0])

        # Update cav_inv_cov (only once).
        if self.lin_response:
            #      and torch.allclose(
            #     cav_inv_cov, torch.zeros_like(cav_inv_cov)
            # ):
            N, dim = spin_mean.size(-2), spin_mean.size(-1)

            J = self.weight()
            S = rearrange(spin_var, "i a b -> a b i")
            V = cav_inv_cov[0]

            A = (
                torch.kron(torch.eye(dim), torch.eye(N))
                - torch.einsum("a c i, i k c d -> a i d k", S, J).reshape(
                    dim * N, dim * N
                )
                + torch.einsum(
                    "a c i, i c d, i k -> a i d k", S, V, torch.eye(N)
                ).reshape(dim * N, dim * N)
            )
            B = rearrange(torch.diag_embed(S), "a b i j -> (a i) (b j)")
            spin_cov = torch.solve(B, A).solution
            spin_cov = rearrange(
                spin_cov, "(a i) (b j) -> a b i j", a=dim, b=dim, i=N, j=N
            )

            # print(torch.linalg.cond(A))

            spin_cov_diag = torch.diagonal(spin_cov, dim1=-2, dim2=-1)  # dim * N
            spin_cov_diag = rearrange(spin_cov_diag, "a b i -> i a b", a=dim, i=N)

            # Find cav var now.
            ones = batched_eye_like(spin_var)
            spin_inv_var = torch.solve(ones, spin_var).solution
            big_lambda = V + spin_inv_var  # (N, d, d)

            A = spin_cov_diag
            B = spin_cov_diag @ big_lambda - batched_eye_like(spin_cov_diag)
            cav_inv_cov = torch.solve(B, A).solution

            print(torch.eig(spin_var[1]))
            print(torch.eig(cav_inv_cov[1]))

            # Restore trivial batch dimension.
            cav_inv_cov = cav_inv_cov.unsqueeze(0).expand(x.shape[0], -1, -1, -1)

            breakpoint()

        # print(next_cav_inv_cov)
        # print(spin_var)
        # print(next_spin_mean)
        # print(self.weight()[0, 1, :, :])
        # breakpoint()

        return self.pack_state([spin_mean, cav_inv_cov])
