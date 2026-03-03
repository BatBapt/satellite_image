import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import configuration as paths
import configuration as cfg

def plot_sample(dataset="train"):
    train_path = os.path.join(paths.MAIN_DATA_PATH, dataset)
    images_path = os.path.join(train_path, "images")
    masks_path = os.path.join(train_path, "gt")

    random_sample = np.random.randint(0, len(os.listdir(images_path)))

    image = plt.imread(os.path.join(images_path, os.listdir(images_path)[random_sample]))
    mask = plt.imread(os.path.join(masks_path, os.listdir(masks_path)[random_sample]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(image)
    ax1.set_title("Image")
    ax1.axis("off")
    ax2.imshow(mask)
    ax2.set_title("Mask")
    ax2.axis("off")
    plt.show()


def compare_predictions(filename, prediction_folder, suffix_models):
    n = len(suffix_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for i, suffixe in enumerate(suffix_models):
        seg_name = f"seg_{suffixe}_{filename}.tif"
        seg_path = os.path.join(prediction_folder, seg_name)

        ax = axes[i]

        masque = Image.open(seg_path)

        ax.imshow(masque, cmap='gray')
        ax.set_title(f"Modèle {suffixe}", fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # plot_sample("train_patches")

    pred_folder = os.path.join(cfg.MAIN_DATA_PATH, "test", "predictions")
    compare_predictions("bellingham35", pred_folder, ["1", "32"])