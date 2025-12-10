import cv2
import numpy as np
from matplotlib import pyplot as plt

# https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
def canny(image,):
    # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open

    # Old ideas, better than just 100 and 200
    # median = np.median(image)
    # absolute_deviation = np.abs(image - median)
    # mad = np.median(absolute_deviation)
    # low = median - mad
    # high = median + mad
    # low = 0.66*median
    # high = 1.33*median

    # OTSU's method, the best
    threshold, result = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    low = 0.5 * threshold
    high = threshold
    edges = cv2.Canny(image, threshold1=low, threshold2=high, L2gradient=True)
    return edges

def main():

    img = cv2.imread("/home/jacob/courses/math663/butterfly.avif", cv2.IMREAD_GRAYSCALE)
    assert img is not None, "Image not found"

    edges = canny(img)

    fig, ax = plt.subplots()
    ax.imshow(edges, cmap='gray')
    ax.set_title("Canny")
    plt.show()

if __name__ == '__main__':
    main()