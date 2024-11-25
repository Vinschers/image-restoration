import numpy as np
import torch
import matplotlib.pyplot as plt

from io_utils import get_img_tensor, get_img_ndarray, get_noise, get_kernel, plot_info, save
from utils import PSNR, fft, convolve, add_dims

from tychonov import analytical_tychonov, tychonov
from total_variation import total_variation

import numpy_gd
import torch_gd


def gradient_descent(F, dF, expected, x0, tau, max_iter=250):
    if isinstance(x0, np.ndarray):
        return numpy_gd.gradient_descent(F, dF, expected, x0, tau, max_iter=max_iter)
    return torch_gd.gradient_descent(F, dF, expected, x0, tau, max_iter=max_iter)


def denoise():
    return lambda x: x


def blur(img, kernel_path):
    kernel = get_kernel(img, kernel_path)
    fft_kernel = fft(kernel)

    fft_kernel = add_dims(fft_kernel, len(img.shape[2:]))

    return lambda x: convolve(x, fft_kernel)


def square_mask(length):
    def apply_mask(x):
        N, M = x.shape[0], x.shape[1]
        y = x.clone()
        y[(N - length) // 2 : (N - length) // 2 + length, (M - length) // 2 : (M - length) // 2 + length, :] = 0
        return y

    return apply_mask


def main():
    img_path = "experiments/bat512.tif"
    noise = 32

    # original_img = get_img_ndarray(img_path)
    original_img = get_img_tensor(img_path, torch.device("mps" if torch.backends.mps.is_available() else "cpu"))

    func = denoise()
    # func = blur(original_img, "blur-kernels/levin6.txt")
    # func = square_mask(32)

    img = func(original_img) + get_noise(original_img, noise)

    analytical_img = analytical_tychonov(img, 1)

    F, dF, tau = tychonov(img, 1, func)
    # F, dF, tau = total_variation(img, 1e-2, 0.003, func)

    restored_img, Y, diffs = gradient_descent(F, dF, analytical_img, img, tau, 100)

    # plot_info(original_img, img, restored_img, Y, diffs)

    save(restored_img, "restored.png")
    plt.plot(Y, label="error", color="blue")
    plt.plot(diffs, label="difference", color="orange")
    plt.title("Difference to solution")
    plt.legend()


if __name__ == "__main__":
    main()
