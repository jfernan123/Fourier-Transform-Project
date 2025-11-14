import cv2
from matplotlib import pyplot as plt

# https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
img = cv2.imread("your-image.jpg", cv2.IMREAD_GRAYSCALE)
assert img is not None, "Image not found"

edges = cv2.Canny(img, threshold1=100, threshold2=200)

fig, ax = plt.subplots()
ax.imshow(edges, cmap='gray')
ax.set_title("Canny")
plt.show()
