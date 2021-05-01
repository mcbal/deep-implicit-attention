import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from deep_implicit_attention.attention import (
    DeepImplicitAttention,
    ExplicitDeepImplicitAttention,
)
from deep_implicit_attention.deq import DEQFixedPoint
from deep_implicit_attention.solvers import anderson


class TestGradients(unittest.TestCase):
    def test_explicit_deep_implicit_attention(self):
        """Run a small network with double precision."""

        num_spins, dim = 11, 3

        for lin_response in [False, True]:
            with self.subTest():
                deq_attn = DEQFixedPoint(
                    ExplicitDeepImplicitAttention(
                        num_spins=num_spins,
                        dim=dim,
                        lin_response=lin_response,
                    ),
                    anderson,
                ).double()

                source = torch.randn(1, num_spins, dim).double() / np.sqrt(dim)

                self.assertTrue(
                    gradcheck(
                        deq_attn,
                        source.requires_grad_(),
                        eps=1e-4,
                        atol=1e-3,
                        check_undefined_grad=False,
                    )
                )

    def test_deep_implicit_attention(self):
        """Run a small network with double precision."""

        num_spins, dim = 11, 3

        for lin_response in [False, True]:
            with self.subTest():
                deq_attn = DEQFixedPoint(
                    DeepImplicitAttention(
                        num_spins=num_spins,
                        dim=dim,
                        lin_response=lin_response,
                    ),
                    anderson,
                ).double()

                source = torch.randn(1, num_spins, dim).double() / np.sqrt(dim)

                self.assertTrue(
                    gradcheck(
                        deq_attn,
                        source.requires_grad_(),
                        eps=1e-4,
                        atol=1e-3,
                        check_undefined_grad=False,
                    )
                )


if __name__ == '__main__':
    unittest.main()
