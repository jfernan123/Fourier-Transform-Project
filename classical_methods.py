import cv2
import numpy as np
import os
from glob import glob
from Utils.load_utils import load_bsds500
from noise import add_gaussian_noise
from matplotlib import pyplot as plt
from metrics import calculate_accuracy, psnr
from canny import canny


root = "../BSDS500/BSDS500"

data = load_bsds500(root)

images = data["images"]["train"]

test = images[22]        # use flowers
print("Image shape:", test.shape)


test_noise = add_gaussian_noise(test, 0, 20, 1)

# classical filters
filtered_gaussian = cv2.GaussianBlur(test_noise, (5,5), 3, 0)
filtered_median = cv2.medianBlur(test_noise, 5)
filtered_bilateral = cv2.bilateralFilter(test_noise, 9, 75, 75)

# compute edges using canny
edges_clean = canny(test)
edges_noisy = canny(test_noise)

edges_gauss = canny(filtered_gaussian)
edges_median = canny(filtered_median)
edges_bilateral = canny(filtered_bilateral)

# compare metrics
noise_accuracy = calculate_accuracy(edges_clean, edges_noisy)
gauss_accuracy = calculate_accuracy(edges_clean, edges_gauss)
median_accuracy = calculate_accuracy(edges_clean, edges_median)
bilateral_accuracy = calculate_accuracy(edges_clean, edges_bilateral)

print("Noise accuracy (%):", noise_accuracy * 100)
print("Gaussian filter accuracy (%):", gauss_accuracy * 100)
print("Median filter accuracy (%):", median_accuracy * 100)
print("Bilateral filter accuracy (%):", bilateral_accuracy * 100)

acc_noisy = calculate_accuracy(edges_clean, edges_noisy)
acc_gauss = calculate_accuracy(edges_clean, edges_gauss)
acc_median = calculate_accuracy(edges_clean, edges_median)
acc_bilateral = calculate_accuracy(edges_clean, edges_bilateral)

psnr_gauss = psnr(test, filtered_gaussian)
psnr_median = psnr(test, filtered_median)
psnr_bilateral = psnr(test, filtered_bilateral)

print("\n=== Edge Accuracy (%) ===")
print("Noisy:", acc_noisy * 100)
print("Gaussian:", acc_gauss * 100)
print("Median:", acc_median * 100)
print("Bilateral:", acc_bilateral * 100)

print("\n=== PSNR vs. Ground Truth ===")
print("Gaussian:", psnr_gauss)
print("Median:", psnr_median)
print("Bilateral:", psnr_bilateral)

# plots
fig, axes = plt.subplots(1, 5, figsize=(14, 5))

axes[0].imshow(edges_noisy, cmap='gray')
axes[0].set_title("Noisy Edges")

axes[1].imshow(edges_gauss, cmap='gray')
axes[1].set_title("Gaussian")

axes[2].imshow(edges_median, cmap='gray')
axes[2].set_title("Median")

axes[3].imshow(edges_bilateral, cmap='gray')
axes[3].set_title("Bilateral")

axes[4].imshow(edges_clean, cmap='gray')
axes[4].set_title("Clean Edges")


for ax in axes:
    ax.axis('off')

plt.savefig("comparison_filters.jpg", dpi=200, bbox_inches='tight')
plt.show()

# ground truth comparison
fig2, axes2 = plt.subplots(1, 5, figsize=(18, 5))

axes2[0].imshow(test, cmap='gray')
axes2[0].set_title("Ground Truth")

axes2[1].imshow(test_noise, cmap='gray')
axes2[1].set_title("Noisy")

axes2[2].imshow(filtered_gaussian, cmap='gray')
axes2[2].set_title(f"Gaussian\nPSNR={psnr_gauss:.2f}")

axes2[3].imshow(filtered_median, cmap='gray')
axes2[3].set_title(f"Median\nPSNR={psnr_median:.2f}")

axes2[4].imshow(filtered_bilateral, cmap='gray')
axes2[4].set_title(f"Bilateral\nPSNR={psnr_bilateral:.2f}")

for ax in axes2:
    ax.axis('off')

plt.savefig("ground_truth_comparison.jpg", dpi=200, bbox_inches='tight')
plt.show()