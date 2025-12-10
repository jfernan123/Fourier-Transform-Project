import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from Utils.load_utils import load_bsds500
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
import cv2
from canny import canny
def pratt_fom(baseline, detected):
    dist = distance_transform_edt(np.invert(baseline))
    alpha = 1/9

    fom = 0

    N, M = detected.shape

    for i in range(0, N):
        for j in range(0, M):
            if detected[i, j] == 255:
                fom += 1.0 / ( 1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(
        np.count_nonzero(baseline),
        np.count_nonzero(detected))

    return fom

def calculate_accuracy(a, b):
    same = np.equal(a, b)
    return np.sum(same) / same.size

def mcc(a, b):
    return matthews_corrcoef(a.flatten(), b.flatten())

def mse(a, b):
    return mean_squared_error(a, b)

def psnr(image_1,image_2):
    # image 1 original image
    # image 2 transformed image
    if np.equal(image_1, image_2).all():
        return 99999

    else:
        MSE = mean_squared_error(image_1,image_2)
        return 20*math.log10(255)- 10*math.log10(MSE)

def calculate_ODS_single(y_true, y_pred, num_thresholds = 100):

    high_vals = np.linspace(0, 255, num_thresholds, dtype=int) 
    low_vals = 0.5 * high_vals
    #thresh_steps = np.linspace(0, 1, num_thresholds)
    y_true = canny(y_true)
    max_f = 0
    for t  in range(len(high_vals)):
        low = low_vals[t]
        high = high_vals[t]
        y_pred = cv2.Canny(y_pred.astype(np.uint8),threshold1=low, threshold2=high, L2gradient=True).astype(int)

        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1) #flatten_data(y_pred=y_pred, y_true=y_true)
        
        f_score = calculate_F_score(y_true, y_pred)
        if f_score > max_f:
            max_f= f_score
    return max_f

def calculate_ODS(y_true, y_pred, num_thresholds):
    high_vals = np.linspace(0, 255, num_thresholds, dtype=int) 
    low_vals = 0.2 * high_vals
    y_true_edges = [canny(y_true_i) for y_true_i in y_true]
    max_f = 0
    for t in range(len(high_vals)):
        low = low_vals[t]
        high = high_vals[t]
        y_pred_edges = [cv2.Canny(y_pred_i.astype(np.uint8), threshold1=low, threshold2=high, L2gradient=True).astype(int) for y_pred_i in y_pred]

        y_true_all = np.concatenate([y_true_i.reshape(-1) for y_true_i in y_true_edges], axis=0)
        y_pred_all = np.concatenate([y_pred_i.reshape(-1) for y_pred_i in y_pred_edges], axis=0)
        f_score = calculate_F_score(y_true_all, y_pred_all)
        if f_score > max_f:
            max_f = f_score
    return max_f


def flatten_data(y_true, y_pred):
    y_pred_all = []
    y_true_all = []
    for i in range(len(y_true)):
        y_true_i = y_true[i]
        y_pred_i = y_pred[i]
        y_pred_i = y_pred_i.astype(np.float32) / 255.0  #Convert between 0 and 1

        y_pred_i = y_pred_i.reshape(-1)
        y_true_i = y_true_i.reshape(-1)
        y_pred_all.append(y_pred_i)
        y_true_all.append(y_true_i)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    return y_true_all, y_pred_all

def calculate_OIS(y_true, y_pred, num_thresholds = 100):

    avg_f = 0
    for i in range(len(y_pred)):
        f = calculate_ODS([y_true[i]], [y_pred[i]], num_thresholds)
        avg_f += f
    avg_f = avg_f / len(y_pred)
    return avg_f

def calculate_precision_recall(y_true, y_pred, threshold = 0):
    y_pred_all = y_pred #(y_pred > threshold).astype(np.uint8)
    true_class = 255
    false_class = 0
    TP = np.logical_and(y_pred_all == true_class, y_true == true_class).sum()
    FP = np.logical_and(y_pred_all == true_class, y_true == false_class).sum()
    FN = np.logical_and(y_pred_all == false_class, y_true == true_class).sum()
    precision = TP / (TP+FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    return precision, recall
def calculate_average_precision(y_true, y_pred):
    y_true_all  = np.concatenate([yt.reshape(-1) for yt in y_true])
    y_score_all = np.concatenate([ys.reshape(-1) for ys in y_pred])

    ap = average_precision_score(y_true_all, y_score_all)
    return ap
def calculate_F_score(y_true,y_pred, threshold = 0):
    prec, recall = calculate_precision_recall(y_true, y_pred, threshold)

    f1_score = 2 * prec * recall / (prec + recall + 1e-8)
    return f1_score


def main():

    bsds_path = "BSR\BSDS500"

    data = load_bsds500(bsds_path)
    images = data["images"]["train"]
    gt  = data["edges"]["train"]
    OIS = calculate_average_precision(y_pred=images, y_true=gt)
    print("OIS", OIS)

if __name__ == "__main__":
    main()
