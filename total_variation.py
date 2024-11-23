from utils import sum, sqrt, squared_norm
from differentiation import grad, laplacian


def F_TV(x, v, epsilon, lambda_, func):
    data_fidelity = sum((func(x) - v) ** 2) / 2
    regularization = lambda_ * sum(sqrt(epsilon**2 + grad(x) ** 2))

    return data_fidelity + regularization


def dF_TV(x, v, epsilon, lambda_):
    data_fidelity = x - v

    regularization = lambda_ * laplacian(x) / sqrt(epsilon ** 2 + squared_norm(grad(x)))

    return data_fidelity - regularization


def total_variation(img, epsilon, lambda_, func):
    def F(x):
        return F_TV(x, img, epsilon, lambda_, func)

    def dF(x):
        return dF_TV(x, img, epsilon, lambda_)

    L = (8 * lambda_) / epsilon
    tau = 1.9 / L
    return F, dF, tau
