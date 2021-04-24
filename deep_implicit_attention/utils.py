import random

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
import numpy as np
import torch


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter_kwargs(kwargs, query):
    return {k.replace(query, ""): v for k, v in kwargs.items() if k.startswith(query)}


##############################################################################
# TENSORS
##############################################################################


def batched_eye_like(X: torch.Tensor):
    return torch.eye(*X.shape[1:], out=torch.empty_like(X))[None, :, :].repeat(
        X.shape[0], 1, 1
    )


##############################################################################
# PLOTTING
##############################################################################


def log_plot(y, title="", xlabel="Iteration", ylabel="Relative residual"):
    plt.semilogy(y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
