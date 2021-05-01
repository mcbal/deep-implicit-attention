import numpy as np
import torch

from deep_implicit_attention.attention import DeepImplicitAttention
from deep_implicit_attention.deq import DEQFixedPoint
from deep_implicit_attention.solvers import anderson


batch_size, num_spins, dim = 4, 64, 16

# Initialize fixed-point wrapper around model system.
deq_attn = DEQFixedPoint(
    DeepImplicitAttention(
        num_spins=num_spins,
        dim=dim,
        weight_init_std=1.0 / np.sqrt(num_spins * dim**2),
        lin_response=True,
    ),
    anderson,
    solver_fwd_max_iter=30,
    solver_fwd_tol=1e-4,
    solver_bwd_max_iter=30,
    solver_bwd_tol=1e-4,
)

# Generate a batch of random data input sources (batches of sets of vectors).
source = torch.randn(batch_size, num_spins, dim) / np.sqrt(dim)

# Solve for fixed-point by acting with deep implicit attention module on data.
out = deq_attn(source, debug=True)
print(out)
