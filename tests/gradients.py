# from torch.autograd import gradcheck

# layer = TanhNewtonImplicitLayer(5, tol=1e-10).double()
# gradcheck(layer, torch.randn(3, 5, requires_grad=True, dtype=torch.double), check_undefined_grad=False)


import numpy as np
import torch
from torch.autograd import gradcheck
from deep_implicit_attention.modules.attention import FixedPointAttention

# run a very small network with double precision, iterating to high precision


num_spins, dim = 5, 3
attention = FixedPointAttention(
    num_spins=num_spins, dim=dim, J_init_std=1.0 / np.sqrt(num_spins * dim)
)
source = torch.randn(1, num_spins, dim).double().requires_grad_()
# (attention(source)[0] * torch.randn_like(source)).sum().backward()

# f = ResNetLayer(2, 2, num_groups=2).double()
# deq = DEQFixedPoint(f, anderson, tol=1e-10, max_iter=500).double()
gradcheck(
    attention,
    source.requires_grad_(True),
    eps=1e-5,
    atol=1e-3,
    raise_exception=True,
    check_undefined_grad=True,
)
