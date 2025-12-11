import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def get_fft_magnitude(image):
    F = fftpack.fft2(image)
    F_shift = fftpack.fftshift(F)

    mag =  np.abs(F_shift)
    power = mag**2

    mag_log = np.log1p(mag)
    power_log = np.log1p(power)

    return mag_log, power_log
