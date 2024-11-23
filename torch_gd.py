import torch

from utils import diff


def gradient_descent(f, df, expected, x0, tau=2e-2, delta=1e-5, max_iter=250):
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

        # print(x.grad - df(x.clone()))

        if diff(x_prev, x).item() < delta:
            break

        iters += 1

    return x.detach(), y, diffs
