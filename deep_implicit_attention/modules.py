from functools import lru_cache

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

from .deq import _DEQModule
from .utils import (
    batched_eye_like,
    make_psd,
    make_symmetric_and_traceless,
    make_traceless,
)


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
      - pairwise coupling matrix weight between spins
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
        num_spins,  # number of vector spin degrees of freedom
        dim,  # vector dimension of degrees of freedom
        weight_init_std=1.0,  # std of random Gaussian weight initialization
        weight_symmetric=True,  # enforce symmetric weight
        weight_training=True,  # turn weight into parameter
        prior_init_std=1.0,  # std(s) of single-site prior(s) ~ N(0, std)
        prior_training=False,  # turn
        lin_response=True,  # toggle linear response correction to mean-field
    ):
        super().__init__()

        self._init_weight(
            num_spins, init_std=weight_init_std, training=weight_training,
        )
        self.weight_symmetric = weight_symmetric
        self.prior_init_std = (
            nn.Parameter(prior_init_std * torch.ones(num_spins, 1))
            if prior_training
            else prior_init_std
        )

        self.lin_response = lin_response

    def _init_weight(self, num_spins, init_std, training):
        """Initialize random coupling matrix."""
        weight = init_std * torch.randn(num_spins, num_spins)
        if training:
            self._weight = nn.Parameter(weight)
        else:
            self.register_buffer("_weight", weight)

    @property
    def weight(self):
        if self.weight_symmetric:
            return make_symmetric_and_traceless(self._weight)
        return make_traceless(self._weight)

    def gibbs_free_energy(self):
        pass

    def get_initial_guess(self, x):
        return [
            torch.zeros_like(x),  # spin_mean
            torch.zeros(
                (x.size(0), x.size(1), 1), device=x.device, dtype=x.dtype
            ),  # cavity_var
        ]

    def _spin_mean_var(self, x, cav_mean, cav_var):
        """
        
        These expressions are obtained from integrating the single-site partition function,
        where a Gaussian prior has been . Inserting a X prior for scalar degrees of freedom
        would give a cosh(...)-expression for the single-site partition function and hence
        a spin expectation value involving tanh(...) (see e.g.)."""
        prefactor = 1.0 / (1.0 / self.prior_init_std ** 2 - cav_var)
        spin_mean = prefactor * (cav_mean + x)
        spin_var = prefactor if self.lin_response else torch.zeros_like(cav_var)
        return spin_mean, spin_var

    def forward(self, z, x, *args):
        spin_mean, cav_var = self.unpack_state(z)

        cav_mean = (
            torch.einsum("n m, b m d -> b n d", self.weight(), spin_mean)
            - cav_var * spin_mean
        )

        next_spin_mean, next_spin_var = self._spin_mean_var(x, cav_mean, cav_var)

        if self.lin_response:
            # Update cav_var (shared across batch so only).
            big_lambda = (cav_var[0] + 1.0 / next_spin_var[0]).squeeze(-1)
            # print(big_lambda.shape)
            X = torch.diag_embed(big_lambda) - self.weight()
            # print(X.shape)
            ones = torch.eye(*X.shape, out=torch.empty_like(X))
            # print(ones.shape)
            out, _ = torch.solve(ones, X)
            next_cav_var = big_lambda - 1.0 / torch.diagonal(out, dim1=-2, dim2=-1)
            # print(next_cav_var)
            # Restore batch.
            bsz = spin_mean.size(0)
            next_cav_var = next_cav_var[None, :, None].repeat(bsz, 1, 1)
        else:
            next_cav_var = cav_var

        return self.pack_state([next_spin_mean, next_cav_var])


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
    """

    def __init__(
        self,
        num_spins,  # number of vector spin degrees of freedom
        dim,  # vector dimension of degrees of freedom
        weight_init_std=1.0,  # std of random Gaussian weight initialization
        weight_symmetric=True,  # enforce symmetric weight
        weight_training=True,  # turn weight into parameter
        prior_init_std=1.0,  # std(s) of single-site prior(s) ~ N(0, std)
        prior_training=False,  # turn
        lin_response=True,  # toggle linear response correction to mean-field
    ):
        super().__init__()

        self._init_weight(
            num_spins, dim, init_std=weight_init_std, training=weight_training,
        )
        self.weight_symmetric = weight_symmetric
        # self.prior_init_std = (
        #     nn.Parameter(prior_init_std * torch.ones(num_spins, dim, dim))
        #     if prior_training
        #     else torch.ones(num_spins, dim, dim)
        # else prior_init_std ** -2
        # * batched_eye_like(torch.ones(num_spins, dim, dim))
        # )

        # bla = 1.0 / (torch.randn(num_spins, dim, dim))  # / (num_spins * dim))
        # self.prior_init_std = 0.5 * (bla + bla.transpose(1, 2))
        # print(self.prior_init_std)

        self.prior_init_std = (
            1.0
            / prior_init_std ** 2
            * batched_eye_like(torch.ones(num_spins, dim, dim))
        )

        # self.prior_init_std = torch.ones(num_spins, dim, dim)

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
        weight = 0.25 * (self._weight + self._weight.permute([1, 0, 3, 2]))
        # weight = 0.5 * (weight + weight.permute([1, 0, 2, 3]))
        mask = batched_eye_like(torch.zeros(dim ** 2, num_spins, num_spins))
        mask = mask.permute([1, 2, 0]).reshape(num_spins, num_spins, dim, dim)
        weight = (1.0 - mask) * weight
        return weight

    def get_initial_guess(self, x):
        # cant initialize cav_var with zero or first prefactor eval is singular (all ones...)
        # cav_var = torch.rand(
        #     (x.size(0), x.size(1), x.size(2), x.size(2)),
        #     device=x.device,
        #     dtype=x.dtype,
        # )
        # cav_var = 0.5 * (cav_var + cav_var.transpose(2, 3))
        # cav_var = (
        #     batched_eye_like(torch.ones(x.size(1), x.size(2), x.size(2)))
        #     .unsqueeze(0)
        #     .repeat(x.size(0), 1, 1, 1)
        # )
        cav_var = torch.zeros(
            (x.size(0), x.size(1), x.size(2), x.size(2)),
            device=x.device,
            dtype=x.dtype,
        )
        return [torch.zeros_like(x), cav_var]  # spin_mean

    def _spin_mean_var(self, x, cav_mean, cav_var):
        """
        
        These expressions are obtained from integrating the single-site partition function,
        where a Gaussian prior has been . Inserting a X prior for scalar degrees of freedom
        would give a cosh(...)-expression for the single-site partition function and hence
        a spin expectation value involving tanh(...) (see e.g.)."""
        # prefactor = 1.0 / (1.0 / (self.prior_init_std ** 2) - cav_var)
        X = self.prior_init_std - cav_var[0]  # (N, d, d)
        ones = batched_eye_like(X)
        prefactor, _ = torch.solve(ones, X)
        # print(prefactor)
        # breakpoint()
        spin_mean = torch.einsum("n d e, b n d -> b n e", prefactor, (cav_mean + x))
        spin_var = prefactor  # .unsqueeze(0)
        return spin_mean, spin_var

    def forward(self, z, x, *args):
        spin_mean, cav_var = self.unpack_state(z)

        cav_mean = torch.einsum(
            "n m d e, b m d -> b n e", self.weight(), spin_mean
        ) - torch.einsum("b n d e, b n d -> b n e", cav_var, spin_mean)

        # print(cav_var.min(), cav_var.mean(), cav_var.max())
        # breakpoint()

        next_spin_mean, next_spin_var = self._spin_mean_var(x, cav_mean, cav_var)

        # Update cav_var (shared across batch so only).
        if self.lin_response:
            N, dim = next_spin_mean.size(-2), next_spin_mean.size(-1)

            J = self.weight()
            # print(torch.norm(J - J.permute([1, 0, 3, 2])))
            # print(J)
            # breakpoint()
            S = rearrange(next_spin_var, "i a b -> a b i")
            V = cav_var[0]
            # print(torch.kron(torch.eye(dim), torch.eye(N)))
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
            covar, _ = torch.solve(B, A)
            covar = rearrange(covar, "(a i) (b j) -> a b i j", a=dim, b=dim, i=N, j=N)

            # print(torch.det(A))
            # print(torch.linalg.cond(A))
            # print(A)
            # breakpoint()

            # print(torch.eig(B))
            # print(torch.eig(rearrange(covar, "a b i j -> (a i) (b j)")))

            chii = torch.diagonal(covar, dim1=-2, dim2=-1)  # dim * N
            chii = rearrange(chii, "a b i -> i a b", a=dim, i=N)

            # print(chii)

            # Find cav var now.
            ones = batched_eye_like(next_spin_var)
            next_spin_varrrr, _ = torch.solve(ones, next_spin_var)
            big_lambda = V + next_spin_varrrr  # (N, d, d)

            A = chii
            B = chii @ big_lambda - batched_eye_like(chii)
            next_cav_var, _ = torch.solve(B, A)

            print(torch.eig(next_spin_var[1]))
            print(torch.eig(next_cav_var[1]))
            # print(next_cav_var[next_cav_var < 0].numel(), next_cav_var.numel())
            # print(next_spin_var[next_spin_var < 0].numel(), next_spin_var.numel())
            # breakpoint()

            # Restore batch.
            next_cav_var = repeat(next_cav_var, "n d e -> b n d e", b=x.size(0))

            # breakpoint()
        else:
            next_cav_var = cav_var  # zeros

        # print(next_cav_var)
        # print(next_spin_var)
        # print(next_spin_mean)
        breakpoint()

        return self.pack_state([next_spin_mean, next_cav_var])
