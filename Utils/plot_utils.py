import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.append(BASE_DIR)
print(BASE_DIR)
import matplotlib.pyplot as plt

from Utils.load_utils import load_bsds500

from canny import canny
from noise import add_gaussian_noise
from noises.fourier.fourier_filter import fourier_denoise
from noises.fourier.fourier_gauss import fourier_gaussian_filter
from noises.fourier.fourier_wiener import fourier_wiener_denoise

def get_fft_magnitude(image):
    F = fftpack.fft2(image)
    F_shift = fftpack.fftshift(F)

    mag =  np.abs(F_shift)
    power = mag**2

    mag_log = np.log1p(mag)
    power_log = np.log1p(power)

    return mag_log, power_log


def plot_spectral_densities(image):
    #I want to plot the spectral densities for each fourier method
    rng = np.random.default_rng(seed=42)

    original_image = image
    #I want first column to be the original. Then second column I want to add the noised image.
    noisy_image = add_gaussian_noise(rng, original_image, 0, 20, 1)
    
    naive_img = fourier_denoise(noisy_image, 40, 50)
    gauss_img = fourier_gaussian_filter(noisy_image, 1)
    wien_img = fourier_wiener_denoise(noisy_image)
    
    # Get magnitude and power spectra for each image
    original_mag, original_power = get_fft_magnitude(original_image)
    noisy_mag, noisy_power = get_fft_magnitude(noisy_image)
    naive_mag, naive_power = get_fft_magnitude(naive_img)
    gauss_mag, gauss_power = get_fft_magnitude(gauss_img)
    wien_mag, wien_power = get_fft_magnitude(wien_img)

    # Create figure with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Add main title
    fig.suptitle('Comparison of Fourier-Based Denoising Methods and Their Power Spectra', 
                 fontsize=20, fontweight='bold', color='black')

    # Row 1: Images
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original Image', color='black', fontsize=18)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_image, cmap='gray')
    axes[0, 1].set_title('Noisy Image', color='black', fontsize=18)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(naive_img, cmap='gray')
    axes[0, 2].set_title('Naive Fourier Denoised', color='black', fontsize=18)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(gauss_img, cmap='gray')
    axes[0, 3].set_title('Fourier Gaussian Filtered', color='black', fontsize=18)
    axes[0, 3].axis('off')

    axes[0, 4].imshow(wien_img, cmap='gray')
    axes[0, 4].set_title('Fourier Wiener Filtered', color='black', fontsize=18)
    axes[0, 4].axis('off')

    # Row 2: Power Spectra (using gray colormap)
    axes[1, 0].imshow(original_power, cmap='gray')
    axes[1, 0].set_title('Original Spectrum', color='black', fontsize=18)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(noisy_power, cmap='gray')
    axes[1, 1].set_title('Noisy Spectrum', color='black', fontsize=18)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(naive_power, cmap='gray')
    axes[1, 2].set_title('Naive Fourier Spectrum', color='black', fontsize=18)
    axes[1, 2].axis('off')

    axes[1, 3].imshow(gauss_power, cmap='gray')
    axes[1, 3].set_title('Fourier Gaussian Spectrum', color='black', fontsize=18)
    axes[1, 3].axis('off')

    axes[1, 4].imshow(wien_power, cmap='gray')
    axes[1, 4].set_title('Fourier Wiener Spectrum', color='black', fontsize=18)
    axes[1, 4].axis('off')

    # Set white background
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.show()

def plot_edge_collage_edges(images, n): 
    # Select images with desired resolution
    rng = np.random.default_rng(seed=42)
    
    # Filter images by resolution first
    filtered_images = [img for img in images if img.shape == (321, 481)]
    
    # Select n images
    num_images = min(n, len(filtered_images))
    selected_images = filtered_images[:num_images]
    
    if num_images == 0:
        print("No images found with resolution (321, 481)")
        return
    
    # Create figure: rows = number of images, columns = 5 edge detection results 
    fig, axes = plt.subplots(num_images, 5, figsize=(30, 4*num_images)) 
     
    if num_images == 1: 
        axes = axes.reshape(1, -1) 
     
     
    col_titles = ['Original Image', 'Original Edges','Naive Denoised Edges',  
                  'Gaussian Filtered Edges', 'Wiener Filtered Edges'] 
    
    for idx, image in enumerate(selected_images):
        original_image = image 
        noisy_image = add_gaussian_noise(rng, original_image, 0, 20, 1) 
         
        naive_img = fourier_denoise(noisy_image, 25, 50) 
        gauss_img = fourier_gaussian_filter(noisy_image, 1) 
        wien_img = fourier_wiener_denoise(noisy_image) 
         
        edges_og = canny(original_image) 
        edges_noisy = canny(noisy_image) 
        edges_naive = canny(naive_img) 
        edges_gauss = canny(gauss_img)     
        edges_wien = canny(wien_img) 
         
        edge_results = [original_image, edges_og, edges_naive, edges_gauss, edges_wien] 
         
        for col_idx, (edges, title) in enumerate(zip(edge_results, col_titles)): 
            axes[idx, col_idx].imshow(edges, cmap='gray') 
            if idx == 0: 
                axes[idx, col_idx].set_title(title, fontsize=18, fontweight='bold', pad=10) 
            axes[idx, col_idx].axis('off') 
     
    fig.patch.set_facecolor('white') 
    
    # Adjust layout to leave space for titles at the top
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(top=0.95)
    plt.show()
if __name__ == "__main__":

    root = "BSR/BSDS500"

    data = load_bsds500(root)

    images = data["images"]["train"] + data["images"]["val"] + data["images"]["test"]
    plot_spectral_densities(images[0])
    plot_edge_collage_edges(images, 5)