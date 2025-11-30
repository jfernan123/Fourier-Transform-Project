import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def show_fft_magnitude(image):
    F = fftpack.fft2(image)
    F_shift = fftpack.fftshift(F) 

    mag =  np.abs(F_shift)               
    power = mag**2 

    mag_log = np.log1p(mag)
    power_log = np.log1p(power) 

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(mag_log, cmap="gray")
    axes[1].set_title("Log magnitude |F|")
    axes[1].axis("off")

    axes[2].imshow(power_log, cmap="gray")
    axes[2].set_title("Log power |F|^2")
    axes[2].axis("off")

    plt.tight_layout()

