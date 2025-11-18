import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_accuracy(a, b):
    same = np.equal(a, b)
    return np.sum(same) / same.size

def psnr(image_1,image_2):
    # image 1 original image
    # image 2 transformed image
    # if the function it's called on itself, PSNR and MSE is zero
    if calculate_accuracy(image_1,image_2) == 1:
        return 0
    else:
        MSE = mean_squared_error(image_1,image_2)
        return 20*np.log10(255)- 10*np.log10(MSE)

# example

# load image as greyscale
#image = cv2.imread('../images/starbunnyy.jpg',cv2.IMREAD_GRAYSCALE)

# show grey image 

#plt.imshow(image)
#plt.title("original image")
#plt.show()


# importing fourier filter from noises directory
# navigating to noises directory
#import os
#os.chdir("..")
#os.chdir("noises/fourier")


# calculating fourier-thresholded image

#import fourier_filter

#filtered = fourier_filter.fourier_denoise(image)
#plt.imshow(filtered)
#plt.title("fourier filtered image")
#plt.show()

# calculating the meansquared error between the two images

#MSE = mean_squared_error(image,filtered)
#print(MSE)
# applying the formula

#psnr = 20*np.log10(255)- 10*np.log10(MSE)
#print(psnr)



