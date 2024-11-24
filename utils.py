import numpy as np
import torch


def sum(x):
    if isinstance(x, np.ndarray):
        return np.sum(x)
    return torch.sum(x)


def sqrt(x):
    if isinstance(x, torch.Tensor):
        return torch.sqrt(x)
    return np.sqrt(x)


def abs(x):
    if isinstance(x, np.ndarray):
        return np.abs(x)
    return torch.abs(x)


def sin(x):
    if isinstance(x, np.ndarray):
        return np.sin(x)
    return torch.sin(x)


def squared_norm(x):
    return sum(x ** 2)


def log10(x):
    if isinstance(x, np.ndarray):
        return np.log10(x)
    return torch.log10(x)


def clip(x, a, b):
    if isinstance(x, np.ndarray):
        return np.clip(x, a, b)
    return torch.clip(x, a, b)


def fft(img):
    if isinstance(img, np.ndarray):
        return np.fft.fft2(img, axes=(0, 1))
    return torch.fft.fft2(img, dim=(0, 1))


def ifft(img):
    if isinstance(img, np.ndarray):
        return np.fft.ifft2(img, axes=(0, 1)).real
    return torch.fft.ifft2(img, dim=(0, 1)).real


def convolve(img, fft_kernel):
    img = ifft(fft(img) * fft_kernel)
    return img


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def diff(img1, img2):
    return (abs(img1 - img2)).max()


def RMSE(img1, img2):
    rmse = sum((img1 - img2) ** 2) / (img1.shape[0] * img1.shape[1])
    return sqrt(rmse)


def PSNR(img1, img2):
    psnr = 20 * log10(img1.max() / RMSE(img1, img2))

    if isinstance(psnr, torch.Tensor):
        return psnr.item()
    return psnr


def add_dims(x, dims):
    for _ in range(dims):
        if isinstance(x, np.ndarray):
            x = np.expand_dims(x, axis=-1)
        else:
            x = x.unsqueeze(-1)

    return x
