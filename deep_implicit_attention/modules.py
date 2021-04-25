import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from .deq import _DEQModule
from .utils import batched_eye_like


class GeneralizedIsingGaussianAdaTAP(_DEQModule):
    """Ising-like vector model with multivariate Gaussian prior over spins.

    Generalization of the application of the adaptive TAP mean-field approach
    from a system of binary/scalar spins to vector spins. Schematically, the
    fixed-point mean-field equations including the Onsager term look like:

        S_i ~ sum_ij J_ij S_j - V_i S_i + x_i

    where the V_i are self-corrections obtained self-consistently and `x_i`
    denote the input injection or magnetic fields applied at site `i`. The linear
    response correction step involves solving a system of equations, leading to
    a complexity ~ O(N^3*d^3). Position-wise feed-forward networks in transformers
    can be thought of as a neural network approximation of this expensive step.
    Mean-field results are obtained by setting V_i = 0.

    Given the couplings between spins and a prior distribution for the single-
    spin partition function, the adaptive TAP framework provides a closed-form
    solution in terms of sets of equations that should be solved self-consistently
    for a fixed point. The algorithm is related to expectation propagation
    (see Section 4.3 in https://arxiv.org/abs/1409.6179) and boils down to
    matching the first and second moments assuming a Gaussian cavity distribution.

    To use this module, wrap it in `modules.DEQFixedPoint`.

    Args:
        num_spins (int):
            Number of (vector) spin degrees of freedom.
        dim (int):
            Internal vector space dimension of the spin degrees of freedom.
        weight_init_std (Optional[float]):
            Standard deviation of random Gaussian weight initialization.
            Defaults to 1.0 / np.sqrt(num_spins * dim ** 2) to ensure |weight| ~ O(1).
        weight_training (bool):
            Allow coupling weights to be trained. (default: `True`).
        lin_response (bool):
            Toggle linear response correction to mean-field (default: `True`).
    """

    def __init__(
        self,
        num_spins,
        dim,
        weight_init_std=None,
        weight_training=True,
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
        self.lin_response = lin_response
        self.spin_prior_inv_var = batched_eye_like(
            torch.zeros(num_spins, dim, dim))

    def _init_weight(self, num_spins, dim, init_std, training):
        """Initialize random coupling matrix."""
        weight = init_std * torch.randn(num_spins, num_spins, dim, dim)
        if training:
            self._weight = nn.Parameter(weight)
        else:
            self.register_buffer('_weight', weight)

    def weight(self):
        """
        Return symmetrized and traceless weight tensor.

        Note:
            This implementation is very inefficient since it recomputes the
            weight tensor every time the function is called during the forward
            pass. Since the couplings stay fixed during a fixed-point iteration,
            the weight tensor should only be computed once and then retrieved
            from a cache.

            As soon as `torch v1.9` is released, this weight should be implemented
            using the new `torch.nn.utils.parametrize` functions, as shown in
            https://pytorch.org/tutorials/intermediate/parametrizations.html.
            The forward pass should take place in the `with parametrize.cached():`
            context manager to cache the weight calculation when first called.
        """
        num_spins, dim = self._weight.size(0), self._weight.size(2)
        # Symmetrize internal dimension.
        weight = 0.5 * (self._weight + self._weight.permute([0, 1, 3, 2]))
        # Symmetrize sites.
        weight = 0.5 * (weight + weight.permute([1, 0, 2, 3]))
        # Make site-site coupling traceless.
        mask = batched_eye_like(torch.zeros(dim ** 2, num_spins, num_spins))
        mask = rearrange(mask, '(a b) i j -> i j a b', a=dim, b=dim)
        weight = (1.0 - mask) * weight
        return weight

    def _initial_guess(self, x):
        """Return initial guess tensors."""
        return [
            torch.zeros_like(x),  # (bsz, N, d)
            torch.zeros((*x.shape, x.shape[-1]),
                        device=x.device, dtype=x.dtype),  # (bsz, N, d, d)
        ]

    def _spin_mean_var(self, x, cav_mean, cav_inv_var):
        """
        Compute spin means and variances from cavity means and inverse covariances matrices.

        Note:
            These expressions are obtained from integrating the single-site partition function,
            with a multivariate Gaussian prior. You should change this function is you want to
            play around with different single-site priors for the spins.
        """
        inv_var = self.spin_prior_inv_var - cav_inv_var  # (N, d, d)
        ones = batched_eye_like(inv_var)
        prefactor = torch.solve(ones, inv_var).solution
        spin_mean = torch.einsum(
            'n d e, b n d -> b n e', prefactor, (cav_mean + x)
        )  # (bsz, N, d)
        spin_var = prefactor  # (N, d, d)
        return spin_mean, spin_var

    def forward(self, z, x, *args):
        """
        Implement adaptive TAP fixed-point iteration step.

        Args:
            z (`torch.Tensor`):
                Current fixed-point state as a batch of big vectors.
            x (`torch.Tensor`):
                Input source injection (data). Shape should match that
                of `spin_mean` in `z` (see `_initial_guess`).

        Returns:
            `torch.Tensor` containing the updated fixed-point state as a batch of big vectors

        """
        spin_mean, cav_inv_var = self.unpack_state(z)

        cav_mean = torch.einsum(
            'n m d e, b m d -> b n e', self.weight(), spin_mean
        ) - torch.einsum('b n d e, b n d -> b n e', cav_inv_var, spin_mean)

        spin_mean, spin_var = self._spin_mean_var(x, cav_mean, cav_inv_var[0])

        if self.lin_response:
            N, dim = spin_mean.size(-2), spin_mean.size(-1)

            J = self.weight()
            S = rearrange(spin_var, 'i a b -> a b i')
            # Get rid of batch: all elements in batch are equal (system property)
            V = cav_inv_var[0]

            A = (
                torch.kron(torch.eye(dim), torch.eye(N))
                - torch.einsum('a c i, i k c d -> a i d k', S, J).reshape(
                    dim * N, dim * N
                )
                + torch.einsum(
                    'a c i, i c d, i k -> a i d k', S, V, torch.eye(N)
                ).reshape(dim * N, dim * N)
            )
            B = rearrange(torch.diag_embed(S), 'a b i j -> (a i) (b j)')
            spin_cov = torch.solve(B, A).solution
            spin_cov = rearrange(
                spin_cov, '(a i) (b j) -> a b i j', a=dim, b=dim, i=N, j=N
            )

            # [DEBUG] check conditioning of system
            # print(torch.linalg.cond(A))

            spin_cov_diag = torch.diagonal(
                spin_cov, dim1=-2, dim2=-1)  # dim * N
            spin_cov_diag = rearrange(
                spin_cov_diag, 'a b i -> i a b', a=dim, i=N)

            # Solve implicit consistency condition for cav_inv_var.
            ones = batched_eye_like(spin_var)
            spin_inv_var = torch.solve(ones, spin_var).solution
            big_lambda = V + spin_inv_var

            A = spin_cov_diag
            B = spin_cov_diag @ big_lambda - batched_eye_like(spin_cov_diag)
            cav_inv_var = torch.solve(B, A).solution

            # [DEBUG] eigvals should be positive (cov matrices should be psd)
            # print(torch.eig(spin_var[0]))  # check for spin 0
            # print(torch.eig(cav_inv_var[0]))  # check for spin 0

            cav_inv_var = cav_inv_var.unsqueeze(
                0).expand(x.shape[0], -1, -1, -1)

        return self.pack_state([spin_mean, cav_inv_var])


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
