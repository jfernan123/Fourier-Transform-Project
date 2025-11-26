import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from Utils.load_utils import load_bsds500
from tqdm import tqdm

def calculate_accuracy(a, b):

    same = np.equal(a, b)
    return np.sum(same) / same.size

def mcc(a, b):
    return matthews_corrcoef(a.flatten(), b.flatten())

def psnr(image_1,image_2):
    # image 1 original image
    # image 2 transformed image
    # if the function it's called on itself, PSNR and MSE is zero
    if calculate_accuracy(image_1,image_2) == 1:
        return 0

    else:
        MSE = mean_squared_error(image_1,image_2)
        return 20*np.log10(255)- 10*np.log10(MSE)

def calculate_ODS(y_true, y_pred, num_thresholds = 100):


    thresh_steps = np.linspace(0, 1, num_thresholds)
    y_true, y_pred = flatten_data(y_pred=y_pred, y_true=y_true)
    max_f = 0
    for t  in tqdm(thresh_steps):

        f_score = calculate_F_score(y_true, y_pred, threshold = t)
        if f_score > max_f:
            max_f= f_score
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
    thresh_steps = np.linspace(0, 1, num_thresholds)
    avg_f = 0
    for i in range(len(y_true)):
        f = calculate_ODS([y_true[i]], [y_pred[i]], num_thresholds)
        avg_f += f
    avg_f = avg_f / len(y_true)
    return avg_f

def calculate_precision_recall(y_true, y_pred, threshold = 0.5):
    y_pred_all = (y_pred > threshold).astype(np.uint8)


    TP = np.logical_and(y_pred_all == 1, y_true == 1).sum()
    FP = np.logical_and(y_pred_all == 1, y_true == 0).sum()
    FN = np.logical_and(y_pred_all == 0, y_true == 1).sum()

    precision = TP / (TP+FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    return precision, recall
def calculate_average_precision(y_true, y_pred):
    y_true_all  = np.concatenate([yt.reshape(-1) for yt in y_true])
    y_score_all = np.concatenate([ys.reshape(-1) for ys in y_pred])

    ap = average_precision_score(y_true_all, y_score_all)
    return ap
def calculate_F_score(y_true,y_pred, threshold = 0.5):
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
