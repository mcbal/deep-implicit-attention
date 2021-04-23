import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from deep_implicit_attention.deq import DEQFixedPoint
from deep_implicit_attention.modules import GeneralizedIsingGaussianAdaTAP
from deep_implicit_attention.solvers import anderson


class TestModules(unittest.TestCase):
    def test_fixed_point_attention(self):
        """Run a small network with double precision, iterating to high precision."""

        num_spins, dim = 12, 8

        deq_attn = DEQFixedPoint(
            GeneralizedIsingGaussianAdaTAP(
                num_spins=num_spins,
                dim=dim,
                weight_init_std=1.0 / np.sqrt(num_spins * dim),
                weight_symmetric=True,
                prior_init_std=1.0,
                lin_response=False,
            ),
            anderson,
            solver_fwd_max_iter=50,
            solver_fwd_tol=1e-4,
            solver_bwd_max_iter=50,
            solver_bwd_tol=1e-4,
        )

        # source O(1), other contribs sum_j O(1/sqrt(N)) * O(1) ~ O(sqrt(N))

        source = torch.randn(1, num_spins, dim).requires_grad_()  # / np.sqrt(dim)
        # source = torch.zeros(1, num_spins, dim).requires_grad_()
        # m = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
        # source = (
        #     m.sample((1, num_spins, dim)).squeeze(-1).requires_grad_()
        # )  # Exponential distributed with rate=1

        # return
        # print(source)  # , torch.norm(source, dim=-1))
        # print(torch.nn.functional.normalize(source, dim=-1, p=2))
        out = deq_attn(source)
        print(out - source)
        # print(source)
        # print(torch.norm(out, dim=-1))  # ORDER 1?
        # print(torch.norm(deq_attn.fun._weight, dim=-1))  # ORDER 1?
        # print(torch.norm(source, dim=-1))  # ORDER 1?


if __name__ == "__main__":
    unittest.main()
