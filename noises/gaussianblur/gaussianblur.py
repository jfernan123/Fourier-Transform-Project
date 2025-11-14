import cv2
import matplotlib.pyplot as plt

#from https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/

# load image from grayscale
image = cv2.imread('your-image.jpg',cv2.IMREAD_GRAYSCALE)

# show image 
plt.imshow(image)
plt.title("original image")
plt.show()



# gaussian blur with sd = 3 and 5 x 5 kernel
filtered = cv2.GaussianBlur(image,(5,5), 3, 0)

plt.imshow(filtered)
plt.title('Gaussian Blurred Image')
plt.show()









