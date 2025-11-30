from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

def load_ground_truth(mat_path):
    data = loadmat(mat_path)
    assert mat_path.endswith(".mat"), "Did not load a .mat file"
    gt_structs = data["groundTruth"][0]

    # first annotator boundary map
    boundary = gt_structs[0]["Boundaries"][0][0]  #This only loads a single annotator. Further expansion for other annotators (Annotator is people who labeled the image, 5 people labeled the same image)
    return boundary

def load_groundTruth(ground_truth_path):
    #Load train data
    data = {"train": [], "test": [], "val": []}
    with tqdm(data.keys(), desc="Loading splits") as pbar:
        for split in pbar:
            pbar.set_postfix(current=split)
            print(ground_truth_path + f"{os.sep}{split}")
            data_paths = os.listdir(ground_truth_path + f"{os.sep}{split}")
            for p in data_paths:
                path = os.path.join(ground_truth_path, split, p)
                data[split].append(load_ground_truth(path))
        #Load Ground truth
            #Load test, train and val
    return data

def load_images(image_path):
    data = {"train": [], "test": [], "val": []}
    with tqdm(data.keys(), desc="Loading splits") as pbar:
        for split in pbar:
            pbar.set_postfix(current=split)
            print(image_path + f"{os.sep}{split}")
            data_paths = os.listdir(image_path + f"{os.sep}{split}")

            for p in data_paths:
                if p.endswith(".db"):
                    continue
                assert p.endswith(".jpg"), "Did not load a .jpg file"

                path = os.path.join(image_path, split, p)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                assert img is not None, "Image not found"

                data[split].append(img)
        #Load Ground truth
            #Load test, train and val
    return data

def load_bsds500(path, images_only = True):
    """
    Path to BSDS500 dataset
    """
    path = os.path.join(path, "data")
    ground_truth_path = os.path.join(path, "groundTruth")
    if not images_only:
        ground_truth_data = load_groundTruth(ground_truth_path)
    images_path = os.path.join(path, "images")

    img_data = load_images(images_path)
    if images_only:
        data = {"images": img_data}
    else:

        data = {"images": img_data, "edges": ground_truth_data}

    return data

def main():
    root = r"..\BSR\BSDS500"

    # Load everything
    data = load_bsds500(root)

    images = data["images"]["train"]
    edges  = data["edges"]["train"]
    print(edges[0])
    print("Loaded:")
    print(f"- {len(images)} training images")
    print(f"- {len(edges)} training edge maps")

    # Plot a few samples
    num_samples = 5
    indices = np.random.choice(len(images), num_samples, replace=False)

    plt.figure(figsize=(12, 4 * num_samples))
    for i, idx in enumerate(indices):
        img  = images[idx]
        edge = edges[idx]

        # Image subplot
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(img)
        plt.title(f"Image #{idx}")
        plt.axis("off")

        # Edge map subplot
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(edge, cmap="gray")
        plt.title(f"Edge Map #{idx}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
