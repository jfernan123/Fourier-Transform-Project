from Utils.load_utils import load_bsds500
from noise import add_gaussian_noise
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from metrics import calculate_accuracy, psnr, mcc
from wavelet_denoising_example import dwt2_denoise, multilevel_denoise
from canny import canny
import numpy as np
import cv2
import pywt

def sobels():
    root = "../BSDS500/BSDS500"

    data = load_bsds500(root)

    images = data["images"]["train"] + data["images"]["val"] + data["images"]["test"]
    plt.rcParams.update({'font.size': 10})
    # Good images:
    # church - 18
    # seastar - 19
    # flowers - 22!
    # shoes - 32
    # others - 28, 30, 35, 41
    # Canny example
    image_0 = images[19]
    front = canny(image_0)
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(image_0,cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(front, cmap = 'gray')
    ax2.set_title('Canny Edge Detection Algorithm')
    plt.savefig('frontpicture.jpg', bbox_inches='tight',dpi=200)


    # gesiha photo 
    image = images[20]
    noised_image = add_gaussian_noise(image, 20, 40, 1)
   
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(image, cmap = 'gray')
    ax1.set_title('Original from BSDS')
    ax2.imshow(noised_image,cmap = 'gray')
    ax2.set_title('Noised Image mu = 20, var = 40, gamma=1')
    plt.savefig('noisecomparison.jpg', bbox_inches='tight',dpi=200)
    plt.show()


    ## iterating through kernel size
    sobels = []
    sobels_denoise = []
    for i in [1,3,5,7]:
        test1 = cv2.Sobel(image,-1, dx=1,dy=0,ksize = i)
        test2 = cv2.Sobel(noised_image, -1, dx=1,dy=0,ksize= i)
        #abs_sobel64f = np.absolute(test)
        #test = np.uint8(abs_sobel64f)
        sobels.append(test1)
        sobels_denoise.append(test2)


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(sobels[0], cmap="gray")
    ax1.set_title("Sobel Kernel Size 1 x 1")
    ax2.imshow(sobels[1], cmap="gray")
    ax2.set_title("Sobel Kernel Size 3 x 3")
    ax3.imshow(sobels[2], cmap="gray")
    ax3.set_title("Sobel Kernel Size 5 x 5")
    ax4.imshow(sobels[3], cmap="gray")
    ax4.set_title("Sobel Kernel Size 7 x 7")
    # ax1.imshow(image, cmap='gray')
    # ax2.imshow(noisy_image, cmap='gray')
    # ax3.imshow(wavelet_denoise, cmap='gray')
    # ax4.imshow(gaussian_denoise, cmap='gray')
    plt.savefig("sobel_comparison.jpg", bbox_inches='tight', dpi=200)
    plt.show()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(sobels_denoise[0], cmap="gray")
    ax1.set_title("Sobel Kernel Size 1 x 1")
    ax2.imshow(sobels_denoise[1], cmap="gray")
    ax2.set_title("Sobel Kernel Size 3 x 3")
    ax3.imshow(sobels_denoise[2], cmap="gray")
    ax3.set_title("Sobel Kernel Size 5 x 5")
    ax4.imshow(sobels_denoise[3], cmap="gray")
    ax4.set_title("Sobel Kernel Size 7 x 7")
    plt.savefig("sobel_noise_comparison.jpg",bbox_inches='tight',dpi=200)
    plt.show()

    cannys = []
    cannys_denoise = [] 
    test1 =  canny(image)
    canny_denoised_image = canny(noised_image)
    sobel_denoised_image = cv2.Sobel(noised_image, -1, dx=1,dy=0,ksize= i)
    fig, ((ax1,ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2,3)
    ax1.imshow(image,cmap='gray')
    ax1.set_title('Original')
    ax2.imshow(sobels[1], cmap = 'gray')
    ax2.set_title('Sobel')
    ax3.imshow(test1, cmap='gray')
    ax3.set_title('Canny')
    plt.savefig('Canny Comparison.jpg',dpi=200)
    ax4.imshow(noised_image,cmap='gray')
    ax4.set_title('Noised Image')
    ax5.imshow(sobel_denoised_image, cmap = 'gray')
    ax5.set_title('Sobel on Noised Image')
    ax6.imshow(canny_denoised_image, cmap = 'gray')
    ax6.set_title('Canny Noised Image')
    plt.savefig('Canny on Comparison.jpg',dpi=200)
    plt.show()
    
    haars = []
    haars_denoise = []
    for j in ["soft","hard","greater"]:
        image2 = dwt2_denoise(image, 'haar', j, 0.5)
        image3 = dwt2_denoise(noised_image, 'haar', j, 0.5)
        haars.append(image2)
        haars_denoise.append(image3)

    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
    ax1.imshow(haars[0],cmap='gray')
    ax1.set_title('Soft T = 0.5')
    ax2.imshow(haars[1],cmap='gray')
    ax2.set_title('Hard T = 0.5')
    ax3.imshow(haars[2],cmap='gray')
    ax3.set_title('Greater T = 0.5')
    ax4.imshow(haars_denoise[0], cmap = 'gray')
    ax4.set_title('Soft T = 0.5 noised')
    ax5.imshow(haars_denoise[1], cmap = 'gray')
    ax5.set_title('Hard T = 0.5 noised')
    ax6.imshow(haars_denoise[2], cmap = 'gray')
    ax6.set_title('Greater T = 0.5 on noised')

    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.savefig('thresholdingcomparison.jpg',dpi=200)
    plt.show()

if __name__ == "__main__":
    sobels()


