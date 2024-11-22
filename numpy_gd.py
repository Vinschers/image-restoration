import numpy as np
from copy import deepcopy

from utils import diff, abs


def gradient_descent(F, grad_F, expected, x0, tau=0.01, delta=1e-5, max_iter=1_000):
    print(f"{tau = }")

    if isinstance(x0, tuple):
        x = np.random.rand(*x0)
    else:
        x = deepcopy(x0)

    base_f = F(x)
    y = []
    diffs = []
    iters = 0

    while iters < max_iter:
        delta_x = tau * (-grad_F(x))

        x += delta_x

        y.append(F(x) / base_f)
        diffs.append(diff(x, expected))

        if (abs(delta_x)).max() < delta:
            break

        iters += 1

        if iters % 100 == 0:
            print(f"{iters = }")
    return x, y, diffs
