import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


from ..solvers.anderson import anderson

BACKWARD_EPS = 2e-10


class DEQFunc(Function):
    """Autograd function to find equilibrium of module using Anderson solver."""

    @staticmethod
    def f(func, z1, u, *args):
        return func(z1, u, *args)

    @staticmethod
    def g(func, z1, u, cutoffs, *args):
        z1_list = DEQFunc.vec2list(z1, cutoffs)
        return DEQFunc.list2vec(DEQFunc.f(func, z1_list, u, *args)) - z1

    @staticmethod
    def list2vec(z1_list):
        bsz = z1_list[0].size(0)
        return torch.cat([elem.reshape(bsz, -1) for elem in z1_list], dim=1)

    @staticmethod
    def vec2list(z1, cutoffs):
        bsz = z1.shape[0]
        z1_list = []
        start_idx, end_idx = 0, cutoffs[0][0] * cutoffs[0][1]
        for i in range(len(cutoffs)):
            z1_list.append(z1[:, start_idx:end_idx].view(bsz, *cutoffs[i]))
            if i < len(cutoffs) - 1:
                start_idx = end_idx
                end_idx += cutoffs[i + 1][0] * cutoffs[i + 1][1]
        return z1_list

    @staticmethod
    def anderson_find_root(func, z1, u, eps, *args):
        z1_est = DEQFunc.list2vec(z1)
        cutoffs = [(elem.size(1), elem.size(2)) for elem in z1]
        max_iter = args[-2]

        def g(x):
            return DEQFunc.g(func, x, u, cutoffs, *args)

        z1_est, diff = anderson(g, z1_est, max_iter=max_iter, tol=eps)

        import matplotlib.pyplot as plt

        # plt.semilogy(diff)
        # plt.xlabel("Iteration")
        # plt.ylabel("Relative residual")
        # plt.show()

        return DEQFunc.vec2list(z1_est.clone().detach(), cutoffs)

    @staticmethod
    def forward(ctx, func, z1, u, *args):
        nelem = sum([el.nelement() for el in z1])
        eps = args[-1] * np.sqrt(nelem)
        ctx.args_len = len(args)
        with torch.no_grad():
            z1_est = DEQFunc.anderson_find_root(func, z1, u, eps, *args)
            return tuple(z1_est)

    @staticmethod
    def backward(ctx, grad_z1):
        grad_args = [None for _ in range(ctx.args_len)]
        return (None, grad_z1, None, *grad_args)


class DEQModule(nn.Module):
    def __init__(self, func, func_copy):
        super(DEQModule, self).__init__()
        self.func = func
        self.func_copy = func_copy

    def forward(self, z1s, us, z0, **kwargs):
        raise NotImplementedError

    class Backward(Function):
        @staticmethod
        def forward(ctx, func_copy, z1, u, *args):
            ctx.save_for_backward(z1)
            ctx.u = u
            ctx.func = func_copy
            ctx.args = args
            return z1.clone()

        @staticmethod
        def backward(ctx, grad):
            # grad should have dimension (bsz x N x dim)
            print(grad.size())
            big_dim = grad.size(1)
            grad = grad.clone()
            (z1,) = ctx.saved_tensors
            u = ctx.u
            args = ctx.args

            cutoffs = [(elem.size(1), elem.size(2)) for elem in z1]
            max_iter = args[-2]

            func = ctx.func
            z1_temp = z1.clone().detach().requires_grad_()
            u_temp = u.clone().detach()
            args_temp = args

            with torch.enable_grad():
                y = DEQFunc.g(func, z1_temp, u_temp, cutoffs, *args_temp)

            def g(x):
                y.backward(x, retain_graph=True)  # Retain for future calls to g
                res = z1_temp.grad + grad
                z1_temp.grad.zero_()
                return res

            eps = BACKWARD_EPS * np.sqrt(big_dim)
            dl_df_est = torch.zeros_like(grad)
            dl_df_est, diff = anderson(g, dl_df_est, max_iter=max_iter, tol=eps)

            import matplotlib.pyplot as plt

            plt.semilogy(diff)
            plt.xlabel("Iteration")
            plt.ylabel("Relative residual")
            plt.show()

            y.backward(torch.zeros_like(dl_df_est), retain_graph=False)

            grad_args = [None for _ in range(len(args))]
            return (None, dl_df_est, None, *grad_args)


class DEQWrapper(DEQModule):
    """Wrapper for DEQModule acting on list of fixed-point variables."""

    def __init__(self, func, func_copy):
        super(DEQWrapper, self).__init__(func, func_copy)

    def forward(self, z0, u, **kwargs):
        """Bla.

        :param z0: list of initial guesses of module's fixed-point variables
        :param u: inputs aka sources aka probes aka data injection (B, N, d)
        """
        model_args = []
        solver_args = [
            kwargs["max_iter"],
            kwargs["tol"],
        ]
        args = model_args + solver_args

        if u is None:
            raise ValueError("Input injection is required.")

        z1 = list(DEQFunc.apply(self.func, z0, u, *args))
        if self.training:
            cutoffs = [(elem.size(1), elem.size(2)) for elem in z1]
            z1 = DEQFunc.list2vec(DEQFunc.f(self.func, z1, u, *args))
            z1 = self.Backward.apply(self.func_copy, z1, u, *args)
            z1 = DEQFunc.vec2list(z1, cutoffs)
        return z1
