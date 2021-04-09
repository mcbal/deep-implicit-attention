import numpy as np
import torch

from deep_implicit_attention.modules.attention import FixedPointAttention


if __name__ == "__main__":
    num_spins, dim = 4, 4
    attention = FixedPointAttention(
        num_spins=num_spins, dim=dim, J_init_std=1.0 / np.sqrt(num_spins * dim)
    )

    source = torch.randn(1, num_spins, dim).requires_grad_(
        True
    )  # , dtype=torch.float64)
    print(source)

    out = attention(source)  # (1, num_spins, dim)
    print(out)
    (out[0] * torch.randn_like(source)).sum().backward()
    # print(response)
