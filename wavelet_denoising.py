import numpy as np
import math

# https://www.sciencedirect.com/topics/computer-science/wavelet-denoising
# orthonormal wavelet to make sure noise is still gaussian

def sd(coeffs):
    # coeffs = coeffs[np.nonzero(coeffs)]
    # Robust estimate of std dev
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    return np.median(np.absolute(coeffs)) / 0.6745

# T = var(noise) / sd(signal)
def bayes_threshold(image, noise_variance):
    # image = signal + noise, where signal and noise are independent
    # so var(image) = var(signal) + var(noise)
    # so sd(signal) = sqrt(var(image) - var(noise))
    # signal_variance = np.var(image) - noise_variance
    signal_variance = np.mean(image*image) - noise_variance
    # Sometimes variance is zero, so make it very small
    # to avoid divide by zero
    signal_variance = max(signal_variance, np.finfo(image.dtype).eps)

    return noise_variance / math.sqrt(signal_variance)

def multilevel_denoise(image, wavelet, mode):
    coeffs = pywt.wavedec2(image, wavelet)

    new_coeffs = [coeffs[0]]

    highest_detail_coeffs = coeffs[-1][-1]
    noise_variance = sd(highest_detail_coeffs)**2

    for (cH, cV, cD) in coeffs[1:]:
        tH = bayes_threshold(cH, noise_variance)
        tV = bayes_threshold(cV, noise_variance)
        tD = bayes_threshold(cD, noise_variance)

        tcH = pywt.threshold(cH, tH, mode)
        tcV = pywt.threshold(cV, tV, mode)
        tcD = pywt.threshold(cD, tD, mode)

        new_coeffs.append((tcH, tcV, tcD))

    image_2 = pywt.waverec2(new_coeffs, wavelet)

    if np.shape(image) != np.shape(image_2):
        image_2 = image_2[0:np.shape(image_2)[0]-1, 0:np.shape(image_2)[1]-1]

    return image_2.clip(0, 255).astype("uint8")
