import numpy as np
import torch

from utils import sin, sum, squared_norm, fft, ifft, normalize, add_dims
from differentiation import grad, laplacian


def L(v):
    N, M = v.shape[:2]

    if isinstance(v, np.ndarray):
        xi = np.pi * np.arange(N)
        zeta = np.pi * np.arange(M)
    else:
        xi = torch.pi * torch.arange(N, device=v.device)
        zeta = torch.pi * torch.arange(M, device=v.device)

    X = sin(xi / M) ** 2
    Y = sin(zeta / N) ** 2

    return 4 * (X.reshape(N, 1) + Y.reshape(1, M))


def u_chapeau(v, lambda_):
    L_matrix = 1 + 2 * lambda_ * L(v)
    L_matrix = add_dims(L_matrix, len(v.shape[2:]))

    return v / L_matrix


def analytical_tychonov(img, lambda_):
    img_fft = fft(img)
    u = u_chapeau(img_fft, lambda_)
    return normalize(ifft(u))


def dF_tychonov(x, v, lambda_):
    return x - v - 2 * lambda_ * laplacian(x)


def F_tychonov(x, v, lambda_, func):
    data_fidelity = sum((func(x) - v) ** 2) / 2
    regularization = lambda_ * squared_norm(grad(x))

    return data_fidelity + regularization


def tychonov(img, lambda_, func):
    def F(x):
        return F_tychonov(x, img, lambda_, func)

    def dF(x):
        return dF_tychonov(x, img, lambda_)

    L = 8 * lambda_
    tau = 0.1 / L
    return F, dF, tau
