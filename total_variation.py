from utils import sum, sqrt, squared_norm
from differentiation import Dx, Dy, grad, laplacian


def F_TV(x, v, epsilon, lambda_, func):
    data_fidelity = sum((func(x) - v) ** 2) / 2
    regularization = lambda_ * sum(sqrt(epsilon ** 2 + Dx(x) ** 2 + Dy(x) ** 2))

    return data_fidelity + regularization


def dF_TV(x, v, epsilon, lambda_):
    return x - v - lambda_ * laplacian(x) / sqrt(epsilon ** 2 + squared_norm(grad(x)))


def total_variation(img, epsilon, lambda_, func):
    def F(x):
        return F_TV(x, img, epsilon, lambda_, func)

    def dF(x):
        return dF_TV(x, img, epsilon, lambda_)

    L = (8 * lambda_) / epsilon
    tau = 1.9 / L
    return F, dF, tau