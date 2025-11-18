import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
from metrics import psnr
from metrics import calculate_accuracy
from sklearn.metrics import mean_squared_error

# assuming has been passed to cv2.imread(.,cv2.IMREAD_GRAYSCALE)

def dwt2_denoise(image, family, threshold, threshold_value):

    if ((family in pywt.families()) and (threshold in ['soft','hard','garrote','greater','less'])) and threshold_value > 0:

        coeffs = pywt.dwt2(image, family)

        CA, (cH, cV, cD) = coeffs

        tCA = pywt.threshold(CA, threshold_value, threshold)

        tcH = pywt.threshold(cH, threshold_value, threshold)
        tcV = pywt.threshold(cV, threshold_value, threshold)
        tcD = pywt.threshold(cD, threshold_value, threshold)

# applying inverse transform with (t)ransformed coeffs

        image2 = pywt.idwt2((tCA,(tcH,tcV,tcD)), family)

        return image2


#fig , ax = plt.subplots(2,2)
#ax[0,0].imshow(CA, cmap = "cividis"); ax[0,0].set_title("Approximation")
#
#
#ax[1,0].imshow(cH, cmap = "cividis") ; ax[1,0].set_title("Horizontal")
#
#ax[0,1].imshow(cD, cmap = "cividis"); ax[0,1].set_title('vertical detail')
#
#ax[1,1].imshow(cD, cmap = "cividis"); ax[1,1].set_title('diagonal')
#plt.show()
#fig , ax = plt.subplots(2,2)
#ax[0,0].imshow(tCA, cmap = "cividis"); ax[0,0].set_title("Approximation")
#
#
#ax[1,0].imshow(tcH, cmap = "cividis") ; ax[1,0].set_title("Horizontal")
#
#ax[0,1].imshow(tcD, cmap = "cividis"); ax[0,1].set_title('vertical detail')
#
#ax[1,1].imshow(tcD, cmap = "cividis"); ax[1,1].set_title('diagonal')
#plt.show()
