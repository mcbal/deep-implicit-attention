import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from .deq import _DEQModule
from .modules import FeedForward
from .utils import batched_eye, batched_eye_like


class DEQMeanFieldAttention(_DEQModule):
    """Deep implicit attention.

    Attention as a fixed-point mean-field response of an Ising-like vector
    model. Schematically, the fixed-point mean-field equations including
    the Onsager self-correction term look like:

        S_i ~ sum_j J_ij S_j - f(S_i) + x_i

    where `f` is a neural network parametrizing the self-correction term for
    every site and `x_i` denote the input injection or magnetic fields applied
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

        spin_mean = torch.einsum(
            'i j c d, b j d -> b i c', self.weight(), spin_mean) + x
        if self.lin_response:
            spin_mean = spin_mean - self.correction(spin_mean)

        return self.pack_state([spin_mean])


class DEQAdaTAPMeanFieldAttention(_DEQModule):
    """Ising-like vector model with multivariate Gaussian prior over spins.

    Generalization of the application of the adaptive TAP mean-field approach
    from a system of binary/scalar spins to vector spins. Schematically, the
    fixed-point mean-field equations including the Onsager term look like:

        S_i ~ sum_j J_ij S_j - V_i S_i + x_i

    where the V_i are self-corrections obtained self-consistently and `x_i`
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
