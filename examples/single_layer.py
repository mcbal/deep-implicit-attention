import argparse

import numpy as np
import torch

from deep_implicit_attention.deq import DEQFixedPoint
from deep_implicit_attention.modules import GeneralizedIsingGaussianAdaTAP
from deep_implicit_attention.solvers import anderson


def deep_implicit_attention(num_spins, dim, no_lin_response, batch_size):

    deq_attn = DEQFixedPoint(
        GeneralizedIsingGaussianAdaTAP(
            num_spins=num_spins,
            dim=dim,
            weight_init_std=1.0 / np.sqrt(num_spins * dim ** 2),
            lin_response=not no_lin_response,
        ),
        anderson,
        solver_fwd_max_iter=30,
        solver_fwd_tol=1e-4,
        solver_bwd_max_iter=30,
        solver_bwd_tol=1e-4,
    )

    # Generate a batch of random data input sources (sets of vectors).
    source = torch.randn(batch_size, num_spins, dim) / np.sqrt(dim)

    # Act with deep implicit attention module.
    out = deq_attn(source, debug=False)
    print(out - source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Implicit Attention: single layer example"
    )
    parser.add_argument(
        "--num-spins",
        type=int,
        default=23,
        help="Batch size for input data sources (default: 23)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=7,
        help="Batch size for input data sources (default: 7)",
    )
    parser.add_argument(
        "--no-lin-response",
        action="store_true",
        default=False,
        help="Disables linear response correction to mean-field result",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for input data sources (default: 4)",
    )
    args = parser.parse_args()
    deep_implicit_attention(**vars(args))
