import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_accuracy(a, b):
    if np.shape(a) != np.shape(b):
        b = b[0:np.shape(b)[0]-1, 0:np.shape(b)[1] -1]
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
