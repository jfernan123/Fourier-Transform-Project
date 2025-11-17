import numpy as np
import cv2

def add_gaussian_noise(image, mean, stddev, gamma):
    noise = np.zeros(image.shape, dtype=np.uint8)
    cv2.randn(noise, mean, stddev)
    noise = (noise * gamma).astype(np.uint8)
    image = cv2.add(image, noise)
    return image