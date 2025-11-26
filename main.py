from Utils.load_utils import load_bsds500
from noise import add_gaussian_noise
from matplotlib import pyplot as plt
from metrics import calculate_accuracy, psnr, mcc
from wavelet_denoising_example import dwt2_denoise, multilevel_denoise
from canny import canny
import cv2
import pywt

def main():
    root = "../BSDS500/BSDS500"

    data = load_bsds500(root)

    images = data["images"]["train"]

    # Good images:
    # church - 18
    # seastar - 19
    # flowers - 22!
    # shoes - 32
    # others - 28, 30, 35, 41
    test = images[22]
    print(test.shape)

    test_noise = add_gaussian_noise(test, 0, 20, 1)

    # filtered = dwt2_denoise(test_noise, "haar", "soft", 10)
    filtered = multilevel_denoise(test_noise, "db2", "soft")

    # filtered = cv2.GaussianBlur(test_noise,(3,3), 0)

    edges = canny(test)
    edges_noise = canny(test_noise)
    edges_smoothed = canny(filtered)

    noise_accuracy = calculate_accuracy(edges, edges_noise)
    smoothed_accuracy = calculate_accuracy(edges, edges_smoothed)

    noise_mcc = mcc(edges, edges_noise)
    smoothed_mcc = mcc(edges, edges_smoothed)

    print(noise_mcc)
    print(smoothed_mcc)

    print(noise_accuracy*100)
    print(smoothed_accuracy*100)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1.imshow(edges, cmap='gray')
    # ax2.imshow(edges_noise, cmap='gray')
    # ax3.imshow(edges_smoothed, cmap='gray')
    ax1.imshow(test, cmap='gray')
    ax2.imshow(test_noise, cmap='gray')
    ax3.imshow(filtered, cmap='gray')
    plt.savefig("comparison.jpg", bbox_inches='tight', dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
