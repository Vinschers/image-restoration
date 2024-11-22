import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import io


from utils import normalize, sum, clip, add_dims


def get_img_ndarray(img_path):
    if not img_path:
        exit(0)

    img = io.imread(img_path)
    return img / 255


def get_img_tensor(img_path, device=None):
    if not img_path:
        exit(0)

    img = torch.tensor(io.imread(img_path), device=device)
    return img / 255


def get_noise(img, strength):
    N, M = img.shape[:2]

    if isinstance(img, np.ndarray):
        noise = (np.random.randn(N, M) * strength) / 255
    else:
        noise = (torch.randn(N, M, device=img.device) * strength) / 255

    noise = add_dims(noise, len(img.shape[2:]))
    return noise


def get_kernel(img, kernel_path):
    N, M = img.shape[:2]

    if isinstance(img, np.ndarray):
        K = np.ones((1, 1))
        kernel = np.zeros((N, M))
    else:
        K = torch.ones((1, 1), device=img.device)
        kernel = torch.zeros((N, M), device=img.device)

    if kernel_path:
        K = np.loadtxt(kernel_path, dtype=np.float32)

        if isinstance(img, torch.Tensor):
            K = torch.tensor(K, device=img.device)

    H, W = K.shape

    kernel[0:H, 0:W] = K / sum(K)

    if isinstance(kernel, np.ndarray):
        return np.roll(kernel, (-H // 2, -W // 2), (0, 1))
    return torch.roll(kernel, (-H // 2, -W // 2), (0, 1))


def plot(img):
    plt.imshow(img)
    plt.show()


def plot_info(original_img, img, restored_img, Y=None, diff=None, save=False):
    _, axs = plt.subplots(2, 2, figsize=(10, 10))


    if isinstance(original_img, torch.Tensor):
        original_img = original_img.cpu().numpy()

    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    if isinstance(restored_img, torch.Tensor):
        restored_img = restored_img.cpu().numpy()

    img = clip(img, 0, 1)
    restored_img = normalize(restored_img)

    if len(original_img.shape) > 2:
        axs[0, 0].imshow(original_img)
        axs[0, 1].imshow(img)
        axs[1, 0].imshow(restored_img)
    else:
        axs[0, 0].imshow(original_img, cmap="gray")
        axs[0, 1].imshow(img, cmap="gray")
        axs[1, 0].imshow(restored_img, cmap="gray")

    if save:
        plt.imsave("moon_mask.png", img)
        plt.imsave("moon_tv_restored.png", restored_img)

    axs[0, 0].axis("off")
    axs[0, 0].set_title("Expected solution")

    axs[0, 1].axis("off")  # Hide axes
    axs[0, 1].set_title("Altered image")

    axs[1, 0].axis("off")  # Hide axes
    axs[1, 0].set_title("Restored image")

    if Y and diff:
        axs[1, 1].plot(Y, label="error", color="blue")
        axs[1, 1].plot(diff, label="difference", color="orange")
        axs[1, 1].set_title("Difference to solution")
        axs[1, 1].legend()

    plt.tight_layout()
    plt.show()