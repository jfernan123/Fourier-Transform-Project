from scipy import fftpack
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append("../..")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.append(BASE_DIR)

import numpy as np
from Utils.load_utils import load_bsds500
def fourier_gaussian_filter(image, sigma):
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape

    F = np.fft.fftn(img)

    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    fy, fx = np.meshgrid(fy, fx, indexing="ij")  # both (H, W)

    f2 = fx**2 + fy**2

    H_gauss = np.exp(-2.0 * (np.pi**2) * (sigma**2) * f2)

    # 3) Apply filter in Fourier domain
    F_filtered = F * H_gauss

    filtered = np.fft.ifftn(F_filtered)
    filtered = np.real(filtered) 

    return filtered

def main():
    data = load_bsds500("BSR\BSDS500")
    img = data["images"]["train"][0]
    filtered = fourier_gaussian_filter(img, sigma=2.0)

    plt.subplot(1, 2, 1); plt.imshow(img, cmap="gray"); plt.title("orig")
    plt.subplot(1, 2, 2); plt.imshow(filtered, cmap="gray"); plt.title("fourier gauss")
    plt.show()

if __name__ == "__main__":
    main()