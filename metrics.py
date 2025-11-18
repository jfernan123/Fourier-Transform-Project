import numpy as np
from sklearn.metrics import mean_squared_error

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
