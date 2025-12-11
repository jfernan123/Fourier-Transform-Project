from Utils.load_utils import load_bsds500
from noise import add_gaussian_noise
from matplotlib import pyplot as plt
from metrics import calculate_accuracy, psnr, mcc
from wavelet_denoising import multilevel_denoise
from canny import canny
import cv2
import pywt
import numpy as np

def main():
    root = "../BSDS500/BSDS500"

    data = load_bsds500(root)

    images = data["images"]["train"] + data["images"]["val"] + data["images"]["test"]

    # Good images:
    # church - 18
    # seastar - 19
    # flowers - 22!
    # shoes - 32
    # others - 28, 30, 35, 41
    image = images[22]

    rng = np.random.default_rng(seed=42)
    noisy_image = add_gaussian_noise(rng, image, 0, 20, 1)
    # noisy_image = image

    wavelet_denoise = multilevel_denoise(noisy_image, "sym9", "soft")
    gaussian_denoise = cv2.GaussianBlur(noisy_image, (5,5), 0)

    cv2.imwrite("noisy-flowers.jpg", noisy_image)
    cv2.imwrite("wavelet-flowers.jpg", wavelet_denoise)

    edges = canny(image)
    edges_noise = canny(noisy_image)
    edges_wavelet = canny(wavelet_denoise)
    edges_gaussian = canny(gaussian_denoise)

    cv2.imwrite("noisy-flower-edges.jpg", edges_noise)
    cv2.imwrite("wavelet-flower-edges.jpg", edges_wavelet)

    noise_accuracy = calculate_accuracy(edges, edges_noise)
    wavelet_accuracy = calculate_accuracy(edges, edges_wavelet)
    gaussian_accuracy = calculate_accuracy(edges, edges_gaussian)

    noise_mcc = mcc(edges, edges_noise)
    wavelet_mcc = mcc(edges, edges_wavelet)
    gaussian_mcc = mcc(edges, edges_gaussian)

    print(noise_mcc)
    print(wavelet_mcc)
    print(gaussian_mcc)

    print(noise_accuracy*100)
    print(wavelet_accuracy*100)
    print(gaussian_accuracy*100)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(edges, cmap='gray')
    ax2.imshow(edges_noise, cmap='gray')
    ax3.imshow(edges_wavelet, cmap='gray')
    ax4.imshow(edges_gaussian, cmap='gray')
    # ax1.imshow(image, cmap='gray')
    # ax2.imshow(noisy_image, cmap='gray')
    # ax3.imshow(wavelet_denoise, cmap='gray')
    # ax4.imshow(gaussian_denoise, cmap='gray')
    plt.savefig("comparison.jpg", bbox_inches='tight', dpi=200)
    plt.show()

    # orthogonal: coif, db, haar, sym

if __name__ == "__main__":
    main()
