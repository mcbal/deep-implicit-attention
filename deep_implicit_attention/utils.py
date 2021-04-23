import random

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
import numpy as np
import torch


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter_kwargs(kwargs, query):
    return {k.replace(query, ""): v for k, v in kwargs.items() if k.startswith(query)}


##############################################################################
# TENSORS
##############################################################################


def make_traceless(X: torch.Tensor):
    """Put zeros on the diagonal of a square matrix."""
    mask = torch.diag(torch.ones(X.size(0), dtype=X.dtype, device=X.device))
    return (1.0 - mask) * X


def make_symmetric(X: torch.Tensor):
    """Symmetrize a square matrix."""
    return 0.5 * (X + X.t())


def make_symmetric_and_traceless(X: torch.Tensor):
    """Symmetrize a square matrix and put its diagonal to zero."""
    assert X.dim() == 2 and X.size(0) == X.size(1)
    return make_traceless(make_symmetric(X))


def batched_eye_like(X: torch.Tensor):
    return torch.eye(*X.shape[1:], out=torch.empty_like(X))[None, :, :].repeat(
        X.shape[0], 1, 1
    )


def make_psd(X: torch.Tensor, tol=1e-2):
    e, v = torch.symeig(X, eigenvectors=True)
    e = torch.max(e, torch.tensor([tol]))
    return torch.matmul(v, torch.matmul(e.diag_embed(), v.transpose(-2, -1)))


##############################################################################
# PLOTTING
##############################################################################


def log_plot(y, xlabel="Iteration", ylabel="Relative residual"):
    plt.semilogy(y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
