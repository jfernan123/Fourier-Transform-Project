import cv2
import matplotlib.pyplot as plt
import numpy as np

# load image
image = cv2.imread('../../images/starbunnyy.jpg',cv2.IMREAD_GRAYSCALE)

# show image 2
plt.imshow(image)
plt.title("original image")
plt.show()



# definiing Kirsch operators
kirschg1 = np.array([[5,5,5], [-3,0,-3],[-3,-3,-3]])
kirschg2 = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
kirschg3 = np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
kirschg4 = np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])

# convolving and showing kirsch filters
kirsches = [kirschg1,kirschg2,kirschg3,kirschg4]

for j in kirsches:
    filtered = cv2.filter2D(image, -1, j)

    plt.imshow(filtered)
    plt.title('kirsch  filtered starbunny')
    plt.show()




# Compute 2D Fourier from image
from numpy.fft import fft2, fftshift

transform = fft2(image)
shifted = fftshift(transform)

plt.plot(transform)
plt.title("fourier transform 2")
plt.show()

plt.plot(shifted)
plt.title("fourier shifted transform")
plt.show()

# magnitude spec

magnitude_spectrum = np.log(np.abs(shifted+1))

# Display starbunny post-FFT

plt.imshow(magnitude_spectrum)
plt.title('magnitude spectrum of starbunny')
plt.show()

import pywt

#apply wavelet transform

#displaying images with color map cividis

coeffs = pywt.dwt2(image, 'haar')

CA, (cH, cV, cD) = coeffs

fig , ax = plt.subplots(2,2)

ax[0,0].imshow(CA, cmap = "cividis"); ax[0,0].set_title("Approximation")


ax[1,0].imshow(cH, cmap = "cividis") ; ax[1,0].set_title("Horizontal")

ax[0,1].imshow(cD, cmap = "cividis"); ax[0,1].set_title('vertical detail')

ax[1,1].imshow(cD, cmap = "cividis"); ax[1,1].set_title('diagonal')
plt.show()

