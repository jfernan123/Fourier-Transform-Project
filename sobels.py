from Utils.load_utils import load_bsds500
from noise import add_gaussian_noise
from matplotlib import pyplot as plt
from metrics import calculate_accuracy, psnr, mcc
from wavelet_denoising_example import dwt2_denoise, multilevel_denoise
from canny import canny
import cv2
import pywt

def sobels():
    root = "../BSDS500/BSDS500"

    data = load_bsds500(root)

    images = data["images"]["train"] + data["images"]["val"] + data["images"]["test"]

    # Good images:
    # church - 18
    # seastar - 19
    # flowers - 22!
    # shoes - 32
    # others - 28, 30, 35, 41

    #eagle photo 
    image = images[21]
    sobels = []
    plt.imshow(image)
    plt.savefig("grayeagle.jpg",bbox_inches='tight',dpi=200)
    plt.title("Grayscale Eagle")
    plt.show()
    ## iterating through kernel size
    for i in [1,3,5,7]:
        test = cv2.Sobel(image,-1, dx=1,dy=0,ksize = i)
        sobels.append(test)


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(sobels[0])
    ax1.set_title("Sobel Kernel Size 1 x 1")
    ax2.imshow(sobels[1])
    ax2.set_title("Sobel Kernel Size 3 x 3")
    ax3.imshow(sobels[2])
    ax3.set_title("Sobel Kernel Size 5 x 5")
    ax4.imshow(sobels[3])
    ax4.set_title("Sobel Kernel Size 7 x 7")
    # ax1.imshow(image, cmap='gray')
    # ax2.imshow(noisy_image, cmap='gray')
    # ax3.imshow(wavelet_denoise, cmap='gray')
    # ax4.imshow(gaussian_denoise, cmap='gray')
    plt.savefig("sobel_comparison.jpg", bbox_inches='tight', dpi=200)
    plt.show()

    # orthogonal: coif, db, haar, sym


