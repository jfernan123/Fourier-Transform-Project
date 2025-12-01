import numpy as np
import cv2

def add_gaussian_noise(image, mean=0.0, stddev=15.0, gamma=1.0):
    img = image.astype(np.float32)
    noise = np.zeros_like(img, dtype=np.float32)
    cv2.randn(noise, mean, stddev)
    img_noisy = img + gamma * noise
    img_noisy = np.clip(img_noisy, 0, 255)
    return img_noisy.astype(np.uint8)
