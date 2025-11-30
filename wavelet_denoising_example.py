import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
from metrics import psnr
from metrics import calculate_accuracy
from sklearn.metrics import mean_squared_error
import numpy as np
import math

# https://www.sciencedirect.com/topics/computer-science/wavelet-denoising

# orthogonal wavelet is best, 'db2', 'sym2'
# orthonormal wavelet to make sure noise is still gaussian

def sd(coeffs):
    # coeffs = coeffs[np.nonzero(coeffs)]
    # Robust estimate of std dev
    # https://en.wikipedia.org/wiki/Median_absolute_deviation
    return np.median(np.absolute(coeffs)) / 0.6745

# def visushrink(cD):
#     # VisuShrink threshold
#     T = sigma * math.sqrt(2 * math.log(cD.size))
#     return T

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
    # noise_variance = np.var(highest_detail_coeffs)

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

    return image_2.astype("uint8")

def dwt2_denoise(image, family, threshold, threshold_value):

    if ((family in pywt.families()) and (threshold in ['soft','hard','garrote','greater','less'])) and threshold_value > 0:

        coeffs = pywt.dwt2(image, family)

        CA, (cH, cV, cD) = coeffs

        tCA = pywt.threshold(CA, threshold_value, threshold)

        tcH = pywt.threshold(cH, threshold_value, threshold)
        tcV = pywt.threshold(cV, threshold_value, threshold)
        tcD = pywt.threshold(cD, threshold_value, threshold)

# applying inverse transform with (t)ransformed coeffs

        image_2 = pywt.idwt2((tCA,(tcH,tcV,tcD)), family)

# returning the image as type u-int8

# resizing the image if need be
        if np.shape(image) != np.shape(image_2):
            image_2 = image_2[0:np.shape(image_2)[0]-1, 0:np.shape(image_2)[1]-1]

        return image_2.astype("uint8")


#fig , ax = plt.subplots(2,2)
#ax[0,0].imshow(CA, cmap = "cividis"); ax[0,0].set_title("Approximation")
#
#
#ax[1,0].imshow(cH, cmap = "cividis") ; ax[1,0].set_title("Horizontal")
#
#ax[0,1].imshow(cD, cmap = "cividis"); ax[0,1].set_title('vertical detail')
#
#ax[1,1].imshow(cD, cmap = "cividis"); ax[1,1].set_title('diagonal')
#plt.show()
#fig , ax = plt.subplots(2,2)
#ax[0,0].imshow(tCA, cmap = "cividis"); ax[0,0].set_title("Approximation")
#
#
#ax[1,0].imshow(tcH, cmap = "cividis") ; ax[1,0].set_title("Horizontal")
#
#ax[0,1].imshow(tcD, cmap = "cividis"); ax[0,1].set_title('vertical detail')
#
#ax[1,1].imshow(tcD, cmap = "cividis"); ax[1,1].set_title('diagonal')
#plt.show()
