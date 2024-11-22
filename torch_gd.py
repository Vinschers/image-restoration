import torch

from utils import diff


def gradient_descent(f, expected, x0, tau=2e-2, delta=1e-5, max_iter=1_000):
    x = x0.clone().requires_grad_(True)
    print(f"{tau = }")
    optim = torch.optim.SGD([x], lr=tau)

    base_f = f(x0).item()
    y = []
    diffs = []
    iters = 0

    while iters < max_iter:
        optim.zero_grad()

        x_prev = x.clone()

        loss = f(x)
        y.append(loss.item() / base_f)
        diffs.append(diff(x, expected).item())

        loss.backward()
        optim.step()

        if diff(x_prev, x).item() < delta:
            break

        iters += 1

    return x.detach(), y, diffs
