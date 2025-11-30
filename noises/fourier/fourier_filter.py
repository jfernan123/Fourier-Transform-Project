from scipy import fftpack
import numpy as np
import sys
import os
sys.path.append("../..")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.append(BASE_DIR)
import matplotlib.pyplot as plt
from Utils.load_utils import load_bsds500

def fourier_denoise(image, percentile=99, K=20):
    # 1. FFT
    F = fftpack.fft2(image)
    F_shift = fftpack.fftshift(F)  # center DC
    H, W = image.shape
    #Obtain the magnitude of the coefficients
    mag = np.abs(F_shift)
    #This operation makes it so that coefficients around the center won't be filted out by the noise removal, thus
    #keeping the quality of the image.
    mag[H//2 - K:H//2 + K, W//2 - K:W//2 + K] = 0
    #Obtain a threshold such that %percentile of the magnitudes are coefficient < threshold
    thresh = np.percentile(mag, percentile)
    mask = mag < thresh  
    F_shift_filtered = F_shift * mask
    #Multiply it by mask and convert back to image
    F_filtered = fftpack.ifftshift(F_shift_filtered)
    image_filtered = np.real(fftpack.ifft2(F_filtered))

    return image_filtered
from Utils.plot_utils import show_fft_magnitude
from noise import add_gaussian_noise

def main():
    data = load_bsds500("BSR\BSDS500")
    img = data["images"]["train"][0]
    img_noised = add_gaussian_noise(img, 0, 30, 1)
    img_denoised = fourier_denoise(img_noised, percentile = 80, K = 0)
    plt.subplot(1, 3, 1); plt.imshow(img, cmap="gray"); plt.title("orig")
    plt.subplot(1, 3, 2); plt.imshow(img_noised, cmap="gray"); plt.title("noise")
    plt.subplot(1, 3, 3); plt.imshow(img_denoised, cmap="gray"); plt.title("denoised")

    show_fft_magnitude(img)
    show_fft_magnitude(img_noised)
    show_fft_magnitude(img_denoised)

    plt.show()

if __name__ == "__main__":
    main()