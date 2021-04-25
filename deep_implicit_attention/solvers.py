"""
Black-box fixed-point solvers and root-finding methods.

TODO:
    - Add additional fixed-point / root solvers (e.g. Broyden)
"""
import torch


def anderson(
    f, x0, m=5, max_iter=50, tol=1e-4, stop_mode='rel', lam=1e-4, beta=1.0, **kwargs
):
    """
    Anderson acceleration for fixed point iteration.

    Args:
        f (`Callable` or `nn.Module`):
            Function to be minimized.
        x0 (`torch.Tensor`):
            A batch of vectors for the initial guess.
        m (`int`):
            Memory of update (window of previous iterations).
        max_iter (`int`):
            Maximum number of iterations (cut-off).
        tol (`float`):
            Tolerance for solution accuracy.
        stop_mode (`str):
            Use relative ("rel) or absolute ('abs') tolerances (default: 'rel').
        lam (`float`):
            Initial guess for diagonal part of low-rank approx of (inverse) Jacobian?
        beta (`float`):
            Mixing parameter (damping for 0 < `beta` < 1, overprojection for `beta` > 1).

    Returns:
            `dict` containing batch of solution vectors and other diagnostics
    """
    bsz, dim = x0.shape
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict = {'abs': [], 'rel': []}
    lowest_dict = {'abs': 1e8, 'rel': 1e8}
    lowest_step_dict = {'abs': 0, 'rel': 0}

    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1: n + 1, 1: n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        alpha = torch.solve(y[:, : n + 1], H[:, : n + 1,
                            : n + 1])[0][:, 1: n + 1, 0]

        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
        diff_dict = {'abs': abs_diff, 'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)

        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = (
                        X[:, k % m].view_as(x0).clone().detach(),
                        gx.clone().detach(),
                    )
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < tol:
            for _ in range(max_iter - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(
                    lowest_dict[alternative_mode])
            break

    out = {
        'result': lowest_xest,
        'lowest': lowest_dict[stop_mode],
        'nstep': lowest_step_dict[stop_mode],
        'prot_break': False,
        'abs_trace': trace_dict['abs'],
        'rel_trace': trace_dict['rel'],
        'tol': tol,
        'max_iter': max_iter,
    }
    X = F = None
    return out
