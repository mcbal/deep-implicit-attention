import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from deep_implicit_attention.deq import DEQFixedPoint
from deep_implicit_attention.modules import IsingGaussianAdaTAP
from deep_implicit_attention.solvers import anderson


class TestGradients(unittest.TestCase):
    def test_fixed_point_attention(self):
        """Run a small network with double precision."""

        num_spins, dim = 11, 3

        deq_attn = DEQFixedPoint(
            IsingGaussianAdaTAP(
                num_spins=num_spins,
                dim=dim,
                weight_init_std=1.0 / np.sqrt(num_spins * dim),
            ),
            anderson,
        ).double()

        source = torch.randn(1, num_spins, dim).double().requires_grad_()

        self.assertTrue(
            gradcheck(
                deq_attn, source, eps=1e-5, atol=1e-3, check_undefined_grad=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
