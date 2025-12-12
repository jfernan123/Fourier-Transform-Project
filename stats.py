from Utils.load_utils import load_bsds500
from noise import add_gaussian_noise
from metrics import *
from wavelet_denoising import multilevel_denoise
from noises.fourier.fourier_filter import fourier_denoise
from noises.fourier.fourier_gauss import fourier_gaussian_filter
from noises.fourier.fourier_wiener import fourier_wiener_denoise
from canny import canny
import cv2
import pywt
import numpy as np
import math
from tqdm import tqdm
from metrics import *
import matplotlib.pyplot as plt

def main():
    root = "BSDS500/BSDS500"

    data = load_bsds500(root)

    images = data["images"]["train"] + data["images"]["val"] + data["images"]["test"]

    all_stats = dict()

    edge_ratios = []

    rng = np.random.default_rng(seed=42)
    denoised_images_all = {}
    for i in tqdm(range(len(images))):
        original_image = images[i]
        noisy_image = add_gaussian_noise(rng, original_image, 0, 20, 1)
        # haar = db1
        # sym{1,2,3} = db{1,2,3}
        denoised_images = {
            #"Classical Gaussian": cv2.GaussianBlur(noisy_image, (5,5), 0, 0),
            #"Classical Median": cv2.medianBlur(noisy_image, 5),
            #"Classical Bilateral": cv2.bilateralFilter(noisy_image, 9, 25, 25),
            "Classical Gaussian": cv2.GaussianBlur(noisy_image, (5,5), 0),
            "Classical Median": cv2.medianBlur(noisy_image, 5),
            "Classical Bilateral": cv2.bilateralFilter(noisy_image, 9, 25, 25),
            "Fourier Naive": fourier_denoise(noisy_image, 30, 50),
            "Fourier Gaussian": fourier_gaussian_filter(noisy_image, 1),
            "Fourier Wiener": fourier_wiener_denoise(noisy_image),
            "Wavelet (haar)": multilevel_denoise(noisy_image, "haar", "soft"),
            "Wavelet (db4)": multilevel_denoise(noisy_image, "db4", "soft"),
            "Wavelet (sym9)": multilevel_denoise(noisy_image, "sym9", "soft"),
            "Wavelet (coif6)": multilevel_denoise(noisy_image, "coif6", "soft"),
        }

        for key in denoised_images.keys():
            if key not in denoised_images_all:
                denoised_images_all[key] = []
            denoised_images_all[key].append(denoised_images[key])
        original_edges = canny(original_image)

        edge_ratio = np.sum(original_edges == 255)/original_edges.size
        edge_ratios.append(edge_ratio)

        for (method, denoised_image) in denoised_images.items():
            #ODS and OIS need to be calculated with the original image, canny is done inside of them
            #Reason why is because ODS chooses accross thresholds
            denoised_edges = canny(denoised_image)
            prec, recall = calculate_precision_recall(original_edges, denoised_edges)
            metrics = {
                # Add other metrics here
                "ACC": calculate_accuracy(original_edges, denoised_edges),
                "MSE": mse(original_image, denoised_image),
                "MCC": mcc(original_edges, denoised_edges),
                "F-Score": calculate_F_score(original_edges, denoised_edges),
                "Precision": prec,
                "Recall": recall,
                "ODS": 0,
                "OIS": 0
                # "FOM": pratt_fom(original_edges, denoised_edges),
            }

            for (metric, stat) in metrics.items():
                key = (method, metric)
                if key not in all_stats:
                    all_stats[key] = list()
                all_stats[key].append(stat)
    avg_stats = { key: np.mean(l) for (key, l) in all_stats.items() }

    #Calculate OIS for each algorithm
    for algo in tqdm(denoised_images_all.keys()):
        denoised_images_algo = denoised_images_all[algo]
        ODS = calculate_ODS(images, denoised_images_algo, num_thresholds= 100)
        OIS = calculate_OIS(images, denoised_images_algo, num_thresholds= 100)
        avg_stats[(algo, "OIS")] = OIS
        avg_stats[(algo, "ODS")] = ODS
    #Calculate ODS for the whole image
    
    for ((method, metric), avg) in avg_stats.items():
        if metric == "MSE":
            psnr = 20*math.log10(255)-10*math.log10(avg)
            print((method, "PSNR"), f"{psnr:.2f}")

        print((method, metric), f"{avg:.3f}")

    print("Avg no edge ratio", 1 - np.mean(edge_ratios))

if __name__ == "__main__":
    main()
