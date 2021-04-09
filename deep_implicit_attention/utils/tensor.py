"""Tensor helper functions."""
import math
import torch

LOG2PI = math.log(2 * math.pi)
INV2PI = (2 * math.pi) ** -1


def solve(P, Q):
    """
    Compute P^-1 Q, where P is a PSD matrix, using the Cholesky factorisation
    """
    L = cho_factor(P)
    return cho_solve(L, Q)


def inv(P):
    """
    Compute the inverse of a PSD matrix using the Cholesky factorisation
    """
    L = cho_factor(P)
    return cho_solve(L, np.eye(P.shape[-1]))


def diag(P):
    """
    a broadcastable version of np.diag, for when P is size [N, D, D]
    """
    return torch.diag_embed(X)


def make_traceless(X: torch.Tensor):
    """Put diagonal of a square matrix to zero in a differentiable way."""
    mask = torch.diag(torch.ones(X.size(0), dtype=X.dtype, device=X.device))
    return (1.0 - mask) * X


def make_symmetric(X: torch.Tensor):
    """Symmetrize a square matrix."""
    return 0.5 * (X + X.t())


def make_symmetric_and_traceless(X: torch.Tensor):
    """Symmetrize a square matrix and put its diagonal to zero."""
    assert X.dim() == 2 and X.size(0) == X.size(1)
    return make_traceless(make_symmetric(X))
