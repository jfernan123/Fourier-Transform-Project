import numpy as np
import cv2

def add_gaussian_noise(rng, image, mean, stddev, gamma):
    img = image.astype(np.float32)
    noise = rng.normal(mean, stddev, image.shape)
    img_noisy = img + gamma * noise
    img_noisy = np.clip(img_noisy, 0, 255)
    return img_noisy.astype(np.uint8)
