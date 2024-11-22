import numpy as np
import torch

from io_utils import get_img_tensor, get_img_ndarray, get_noise, get_kernel, plot_info
from utils import fft, convolve, add_dims

from tychonov import analytical_tychonov, tychonov
from total_variation import total_variation

import numpy_gd
import torch_gd


def gradient_descent(F, dF, expected, x0, tau):
    if isinstance(x0, np.ndarray):
        return numpy_gd.gradient_descent(F, dF, expected, x0, tau)
    return torch_gd.gradient_descent(F, expected, x0, tau)


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
    img_path = "experiments/lena.tif"
    noise = 4
    lambda_ = 0.01

    print(f"lambda = {lambda_}")

    # original_img = get_img_ndarray(img_path)
    original_img = get_img_tensor(img_path, torch.device("mps" if torch.backends.mps.is_available() else "cpu"))

    # func = denoise()
    func = blur(original_img, "blur-kernels/levin1.txt")
    # func = square_mask(32)

    img = func(original_img) + get_noise(original_img, noise)

    # analytical_img = analytical_tychonov(img, lambda_)

    # F, dF, tau = tychonov(img, lambda_, func)
    F, dF, tau = total_variation(img, 1e-2, lambda_, func)

    restored_img, Y, diffs = gradient_descent(F, dF, original_img, img, tau)

    plot_info(original_img, img, restored_img, Y, diffs)


if __name__ == "__main__":
    main()
