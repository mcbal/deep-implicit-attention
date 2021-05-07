from functools import partial

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

from .deq import _DEQModule
from .modules import FeedForward
from .utils import batched_eye, batched_eye_like


class DEQMLPMixerAttention(_DEQModule):
    """A deep equilibrium version of MLP-Mixer transformer attention:

        S_i = g({S_j}) - f(S_i) + X_i

    where `g` is an MLP acting across the sequence dimension instead of
    the feature dimension (so across patches). The network `f` acts
    across the feature dimension (so individually on every sequence).

    Compared to a vanilla softmax attention transformer module, the
    sum over couplings has been "amortized" and parametrized by an MLP.
    The fixed-point variables S_i's are also fed straight into the
    feed-forward self-correction term. One could feed `spin_mean_mf`
    instead to fully mimic the residual connection in the explicit
    MLP-Mixer architecture.

    Note:
        To use this module, wrap it in `modules.DEQFixedPoint`.

    Paper:
        https://arxiv.org/abs/2105.02723
        https://arxiv.org/abs/2105.01601
    """

    def __init__(
        self,
        num_spins,
        dim,
        lin_response=True,
    ):
        super().__init__()

        self.sum_over_couplings = FeedForward(
            num_spins, dense=partial(nn.Conv1d, kernel_size=1)
        )  # no dropout

        if lin_response:
            self.correction = FeedForward(dim, dense=nn.Linear)  # no dropout
        self.lin_response = lin_response

    def _initial_guess(self, x):
        """Return initial guess tensors."""
        bsz, N, d = x.shape
        return [torch.zeros((bsz, N, d), device=x.device, dtype=x.dtype)]

    def forward(self, z, x, *args):
        spin_mean, = self.unpack_state(z)

        # Apply sum-over-couplings amortization MLP and add source term.
        spin_mean_mf = self.sum_over_couplings(spin_mean) + x

        # Add parametrized self-correction term. Change `spin_mean`
        # to `spin_mean_mf` below to mimic explicit architecture.
        if self.lin_response:
            spin_mean = spin_mean_mf - self.correction(spin_mean)

        return self.pack_state([spin_mean])


class DEQVanillaSoftmaxAttention(_DEQModule):
    """A deep equilibrium version of vanilla softmax transformer attention:

        S_i = sum_j J_ij S_j - f(S_i) + X_i

    where

        J_ij = [softmax(X W_Q W_K^T X^T / sqrt(dim))]_ij

    Compared to the explicit vanilla softmax attention transformer module,
    there's no values and the fixed-point variables S_i's are fed straight
    into the feed-forward self-correction term.

    Note:
        To use this module, wrap it in `modules.DEQFixedPoint`.

    Paper:
        https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        num_spins,
        dim,
        heads=1,
        dim_head=None,
        scale=None,
        lin_response=True,
    ):
        super().__init__()

        dim_head = dim_head if dim_head is not None else (dim // heads)
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.heads = heads
        self.scale = scale if scale is not None else dim_head ** -0.5

        if lin_response:
            self.correction = FeedForward(dim)  # no dropout
        self.lin_response = lin_response

    def _initial_guess(self, x):
        """Return initial guess tensors."""
        bsz, N, d = x.shape
        return [torch.zeros((bsz, N, d), device=x.device, dtype=x.dtype)]

    def forward(self, z, x, *args):
        spin_mean, = self.unpack_state(z)
        mask = args[0] if len(args) > 0 else None

        # Get queries, keys, and number of heads.
        q, k, h = self.to_q(x), self.to_k(x), self.heads

        # Reshape head dimension into batch for queries, keys, and spin_means.
        q, k, spin_mean_heads = map(
            lambda t: rearrange(
                t, 'b n (h d) -> (b h) n d', h=h), (q, k, spin_mean)
        )

        # Compute scaled queries/keys overlap.
        scaled_overlap = torch.einsum(
            'b i d, b j d -> b i j', q, k) * self.scale

        # Optional masking.
        if mask is not None:
            max_neg_value = -torch.finfo(scaled_overlap.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            scaled_overlap.masked_fill_(~mask, max_neg_value)

        # Sum over softmax of scaled overlap, for all heads.
        sum_over_couplings_heads = torch.einsum(
            'b i j, b j d -> b i d',
            scaled_overlap.softmax(dim=-1),
            spin_mean_heads
        )

        # Merge heads again and add source term (~ residual connection).
        spin_mean_mf = self.to_out(
            rearrange(
                sum_over_couplings_heads,
                '(b h) n d -> b n (h d)', h=h
            )) + x

        # Add parametrized self-correction term. Change `spin_mean`
        # to `spin_mean_mf` below to mimic explicit architecture.
        if self.lin_response:
            spin_mean = spin_mean_mf - self.correction(spin_mean)

        return self.pack_state([spin_mean])


class DEQMeanFieldAttention(_DEQModule):
    """Deep implicit attention.

    Attention as a fixed-point mean-field response of an Ising-like vector
    model. Schematically, the fixed-point mean-field equations including
    the Onsager self-correction term look like:

        S_i = sum_j J_ij S_j - f(S_i) + X_i

    where `f` is a neural network parametrizing the self-correction term for
    every site and `X_i` denote the input injection or magnetic fields applied
    at site `i`. Mean-field results are obtained by dropping the self-
    correction term. This all looks a lot like a transformer.

    Note:
        To use this module, wrap it in `modules.DEQFixedPoint`.

    Args:
        num_spins (int):
            Number of (vector) spin degrees of freedom.
        dim (int):
            Internal vector space dimension of the spin degrees of freedom.
        weight_init_std (Optional[float]):
            Standard deviation of random Gaussian weight initialization.
            Defaults to 1.0 / np.sqrt(num_spins * dim ** 2) to ensure that
            norm of tensor |weight| ~ O(1).
        weight_training (bool):
            Allow coupling weights to be trained. (default: `True`).
        weight_sym_internal (bool):
            Symmetrize internal indices of weight tensor. (default: `False`).
        weight_sym_sites (bool):
            Symmetrize site indices of weight tensor. (default: `False`).
        lin_response (bool):
            Toggle linear response correction to mean-field (default: `True`).

    Blog:
        https://mcbal.github.io/post/deep-implicit-attention-a-mean-field-theory-perspective-on-attention-mechanisms/
    """

    def __init__(
        self,
        num_spins,
        dim,
        weight_init_std=None,
        weight_training=True,
        weight_sym_internal=False,
        weight_sym_sites=False,
        lin_response=True,
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
        self.weight_sym_internal = weight_sym_internal
        self.weight_sym_sites = weight_sym_sites

        if lin_response:
            self.correction = FeedForward(dim)  # no dropout
        self.lin_response = lin_response

    def _init_weight(self, num_spins, dim, init_std, training):
        """Initialize random coupling matrix."""
        weight = torch.zeros(num_spins, num_spins, dim,
                             dim).normal_(0, init_std)
        if training:
            self._weight = nn.Parameter(weight)
        else:
            self.register_buffer('_weight', weight)

    def weight(self):
        """Return symmetrized and traceless weight tensor."""
        num_spins, dim = self._weight.size(0), self._weight.size(2)
        weight = self._weight
        if self.weight_sym_internal:
            weight = 0.5 * (weight + weight.permute([0, 1, 3, 2]))
        if self.weight_sym_sites:
            weight = 0.5 * (weight + weight.permute([1, 0, 2, 3]))
        mask = batched_eye(dim ** 2, num_spins,
                           device=weight.device, dtype=weight.dtype)
        mask = rearrange(mask, '(a b) i j -> i j a b', a=dim, b=dim)
        weight = (1.0 - mask) * weight
        return weight

    def count_params(self):
        num_spins, dim = self._weight.size(0), self._weight.size(2)
        site_factor = 0.5*num_spins * \
            (num_spins-1) if self.weight_sym_sites else num_spins*(num_spins-1)
        internal_factor = 0.5*dim * \
            (dim+1) if self.weight_sym_internal else dim**2
        return site_factor*internal_factor

    def _initial_guess(self, x):
        """Return initial guess tensors."""
        bsz, N, d = x.shape
        return [torch.zeros((bsz, N, d), device=x.device, dtype=x.dtype)]

    def forward(self, z, x, *args):
        """
        Implement deep implicit attention mean-field fixed-point iteration.

        Args:
            z (`torch.Tensor`):
                Current fixed-point state as a batch of big vectors.
            x (`torch.Tensor`):
                Input source injection (data). Shape should match that
                of `spin_mean` in `z` (see `_initial_guess`).

        Returns:
            `torch.Tensor` with updated fixed-point state as batch of vectors

        """
        spin_mean, = self.unpack_state(z)

        spin_mean_mf = torch.einsum(
            'i j c d, b j d -> b i c', self.weight(), spin_mean) + x

        if self.lin_response:
            spin_mean = spin_mean_mf - self.correction(spin_mean)

        return self.pack_state([spin_mean])


class DEQAdaTAPMeanFieldAttention(_DEQModule):
    """Ising-like vector model with multivariate Gaussian prior over spins.

    Generalization of the application of the adaptive TAP mean-field approach
    from a system of binary/scalar spins to vector spins. Schematically, the
    fixed-point mean-field equations including the Onsager term look like:

        S_i ~ sum_j J_ij S_j - V_i S_i + X_i

    where the V_i are self-corrections obtained self-consistently and `X_i`
    denote the input injection or magnetic fields applied at site `i`. The
    linear response correction step involves solving a system of equations,
    leading to a complexity ~ O(N^3*d^3). Mean-field results are obtained
    by setting V_i = 0.

    Given the couplings between spins and a prior distribution for the single-
    spin partition function, the adaptive TAP framework provides a closed-form
    solution in terms of sets of equations that should be solved for a fixed
    point. The algorithm is related to expectation propagation (see Section
    4.3 in https://arxiv.org/abs/1409.6179) and boils down to matching the
    first and second moments assuming a Gaussian cavity distribution.

    Note:
        To use this module, wrap it in `modules.DEQFixedPoint`.

    Args:
        num_spins (int):
            Number of (vector) spin degrees of freedom.
        dim (int):
            Internal vector space dimension of the spin degrees of freedom.
        weight_init_std (Optional[float]):
            Standard deviation of random Gaussian weight initialization.
            Defaults to 1.0 / np.sqrt(num_spins * dim ** 2) to ensure that
            norm of tensor |weight| ~ O(1).
        weight_training (bool):
            Allow coupling weights to be trained. (default: `True`).
        weight_sym_internal (bool):
            Symmetrize internal indices of weight tensor. (default: `True`).
        weight_sym_sites (bool):
            Symmetrize site indices of weight tensor. (default: `True`).
        lin_response (bool):
            Toggle linear response correction to mean-field (default: `True`).

    Blog:
        https://mcbal.github.io/post/deep-implicit-attention-a-mean-field-theory-perspective-on-attention-mechanisms/
    """

    def __init__(
        self,
        num_spins,
        dim,
        weight_init_std=None,
        weight_training=True,
        weight_sym_internal=True,
        weight_sym_sites=True,
        lin_response=True,
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
        self.weight_sym_internal = weight_sym_internal
        self.weight_sym_sites = weight_sym_sites

        self.register_buffer(
            'spin_prior_inv_var',
            batched_eye_like(
                torch.zeros(num_spins, dim, dim))
        )

        self.lin_response = lin_response

    def _init_weight(self, num_spins, dim, init_std, training):
        """Initialize random coupling matrix."""
        weight = torch.zeros(num_spins, num_spins, dim,
                             dim).normal_(0, init_std)
        if training:
            self._weight = nn.Parameter(weight)
        else:
            self.register_buffer('_weight', weight)

    def weight(self):
        """
        Return symmetrized and traceless weight tensor.

        Note:
            This implementation is very inefficient since it stores N^2*d^2
            parameters but only needs N*(N-1)*d*(d+1)/4. Also look into new
            torch parametrization functionality:
            https://pytorch.org/tutorials/intermediate/parametrizations.html
        """
        num_spins, dim = self._weight.size(0), self._weight.size(2)
        weight = self._weight
        if self.weight_sym_internal:
            weight = 0.5 * (weight + weight.permute([0, 1, 3, 2]))
        if self.weight_sym_sites:
            weight = 0.5 * (weight + weight.permute([1, 0, 2, 3]))
        mask = batched_eye(dim ** 2, num_spins,
                           device=weight.device, dtype=weight.dtype)
        mask = rearrange(mask, '(a b) i j -> i j a b', a=dim, b=dim)
        weight = (1.0 - mask) * weight
        return weight

    def count_params(self):
        num_spins, dim = self._weight.size(0), self._weight.size(2)
        site_factor = 0.5*num_spins * \
            (num_spins-1) if self.weight_sym_sites else num_spins*(num_spins-1)
        internal_factor = 0.5*dim * \
            (dim+1) if self.weight_sym_internal else dim**2
        return site_factor*internal_factor

    def _initial_guess(self, x):
        """Return initial guess tensors."""
        bsz, N, d = x.shape
        return [torch.zeros((bsz, N, d), device=x.device, dtype=x.dtype),
                torch.zeros((bsz, N, d, d), device=x.device, dtype=x.dtype)]

    def _spin_mean_var(self, x, cav_mean, cav_var):
        """
        Compute spin means and variances from cavity means and variances.

        Note:
            These expressions are obtained from integrating the single-site
            partition function with a multivariate Gaussian prior. You should
            change this function is you want to play around with different
            single-site priors for the spins.
        """
        inv_var = self.spin_prior_inv_var - cav_var
        prefactor = torch.solve(batched_eye_like(inv_var), inv_var).solution
        spin_mean = torch.einsum(
            'i d e, b i d -> b i e', prefactor, (cav_mean + x)
        )
        spin_var = prefactor
        return spin_mean, spin_var

    def forward(self, z, x, *args):
        """
        Implement adaptive TAP fixed-point iteration step.

        Note:
            The linear response actually does too much work for this module's
            default choice of spin priors. In particular, the intermediate
            `big_lambda` is always a batch of identity matrices.

            WARNING: This module is very slow, especially the backward pass.

        Args:
            z (`torch.Tensor`):
                Current fixed-point state as a batch of big vectors.
            x (`torch.Tensor`):
                Input source injection (data). Shape should match that
                of `spin_mean` in `z` (see `_initial_guess`).

        Returns:
            `torch.Tensor` with updated fixed-point state as batch of vectors

        """
        spin_mean, cav_var = self.unpack_state(z)

        weight = self.weight()

        cav_mean = torch.einsum(
            'i j d e, b j e -> b i d', weight, spin_mean
        ) - torch.einsum('b i d e, b i d -> b i e', cav_var, spin_mean)

        spin_mean, spin_var = self._spin_mean_var(x, cav_mean, cav_var[0])

        if self.lin_response:
            N, dim = spin_mean.shape[-2], spin_mean.shape[-1]

            V = cav_var[0]
            S = rearrange(spin_var, 'i a b -> a b i')
            J = weight

            A = (
                torch.kron(torch.eye(dim, dtype=x.dtype, device=x.device),
                           torch.eye(N, dtype=x.dtype, device=x.device))
                - torch.einsum('a c i, i k c d -> a i d k', S, J).reshape(
                    dim * N, dim * N
                )
                + torch.einsum(
                    'a c i, i c d, i k -> a i d k', S, V, torch.eye(
                        N, dtype=x.dtype, device=x.device)
                ).reshape(dim * N, dim * N)
            )
            B = rearrange(torch.diag_embed(S), 'a b i j -> (a i) (b j)')
            spin_cov = torch.solve(B, A).solution
            spin_cov = rearrange(
                spin_cov, '(a i) (b j) -> a b i j', a=dim, b=dim, i=N, j=N
            )

            # [DEBUG] check conditioning of system
            # print(torch.linalg.cond(A))

            spin_cov_diag = torch.diagonal(spin_cov, dim1=-2, dim2=-1)
            spin_cov_diag = rearrange(spin_cov_diag, 'a b i -> i a b')

            ones = batched_eye_like(spin_var)
            spin_inv_var = torch.solve(ones, spin_var).solution
            big_lambda = V + spin_inv_var
            A = spin_cov_diag
            B = spin_cov_diag @ big_lambda - batched_eye_like(spin_cov_diag)
            cav_var = torch.solve(B, A).solution

            # [DEBUG] eigvals should be positive (cov matrices should be psd)
            # print(torch.eig(spin_var[0]))  # check for spin 0
            # print(torch.eig(cav_var[0]))  # check for spin 0

            cav_var = cav_var.unsqueeze(
                0).expand(x.shape[0], -1, -1, -1)

        return self.pack_state([spin_mean, cav_var])
