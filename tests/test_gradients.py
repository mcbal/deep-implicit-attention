import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from deep_implicit_attention.deq import DEQFixedPoint
from deep_implicit_attention.modules import GeneralizedIsingGaussianAdaTAP
from deep_implicit_attention.solvers import anderson


class TestGradients(unittest.TestCase):
    def test_fixed_point_attention(self):
        """Run a small network with double precision."""

        num_spins, dim = 11, 3

        for lin_response in [False, True]:
            with self.subTest():
                deq_attn = DEQFixedPoint(
                    GeneralizedIsingGaussianAdaTAP(
                        num_spins=num_spins,
                        dim=dim,
                        weight_init_std=1.0 / np.sqrt(num_spins * dim ** 2),
                        lin_response=lin_response,
                    ),
                    anderson,
                ).double()

                source = torch.randn(1, num_spins, dim).double() / np.sqrt(dim)

                self.assertTrue(
                    gradcheck(
                        deq_attn,
                        source.requires_grad_(),
                        eps=1e-5,
                        atol=1e-3,
                        check_undefined_grad=False,
                    )
                )


if __name__ == "__main__":
    unittest.main()
