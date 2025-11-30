from Utils.load_utils import load_bsds500
from noise import add_gaussian_noise
from metrics import calculate_accuracy, psnr, mcc
from wavelet_denoising_example import multilevel_denoise
from canny import canny
import cv2
import pywt
import numpy as np
from tqdm import tqdm

def main():
    root = "../BSDS500/BSDS500"

    data = load_bsds500(root)

    images = data["images"]["train"] + data["images"]["val"] + data["images"]["test"]

    all_stats = dict()

    # for i in tqdm(range(1)):
    for i in tqdm(range(len(images))):
        original_image = images[i]
        noisy_image = add_gaussian_noise(original_image, 0, 20, 1)

        # haar = db1
        # sym{1,2,3} = db{1,2,3}
        denoised_images = {
            # Add other denoising methods here
            #Classical methods
            "Classical Gaussian": cv2.GaussianBlur(noisy_image, (5,5), 3, 0),
            "Classical Median": cv2.medianBlur(noisy_image, 5),
            "Classical Bilateral": cv2.bilateralFilter(noisy_image, 9, 75, 75),
            #Wavelet methods
            "Wavelet (haar)": multilevel_denoise(noisy_image, "haar", "soft"),
            "Wavelet (db2)": multilevel_denoise(noisy_image, "db2", "soft"),
            "Wavelet (db3)": multilevel_denoise(noisy_image, "db3", "soft"),
            "Wavelet (db4)": multilevel_denoise(noisy_image, "db4", "soft"),
            "Wavelet (db5)": multilevel_denoise(noisy_image, "db5", "soft"),
            # "Wavelet (sym2)": multilevel_denoise(noisy_image, "sym2", "soft"),
            "Wavelet (coif2)": multilevel_denoise(noisy_image, "coif2", "soft")
        }

        original_edges = canny(original_image)

        for (method, denoised_image) in denoised_images.items():
            denoised_edges = canny(denoised_image)

            metrics = {
                # Add other metrics here
                "PSNR": psnr(original_image, denoised_image),
                "MCC": mcc(original_edges, denoised_edges),
            }

            for (metric, stat) in metrics.items():
                key = (method, metric)
                if key not in all_stats:
                    all_stats[key] = list()
                all_stats[key].append(stat)

    avg_stats = { key: np.mean(l) for (key, l) in all_stats.items() }

    for (key, avg) in avg_stats.items():
        print(key, avg)

if __name__ == "__main__":
    main()
