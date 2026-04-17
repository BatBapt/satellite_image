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
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))

    if n == 1:
        axes = [axes]

    for i, suffixe in enumerate(suffix_models):
        seg_name = f"seg_{suffixe}_{filename}.tif"
        seg_path = os.path.join(prediction_folder, seg_name)

        ax = axes[i]

        try:
            masque = Image.open(seg_path)
        except FileNotFoundError:
            print(f"File not found: {seg_path}")
            print("Make sure to run the prediction script to generate the segmentation masks before visualizing.")
            return

        ax.imshow(masque, cmap='gray')
        ax.set_title(f"Modèle {suffixe}", fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("plots/comparison_predictions.png")
    plt.show()


def plot_val_pred(gt_folder, pred_folder, nb_sample=5, best_model="33"):
    fig, axes = plt.subplots(nb_sample, 2, figsize=(10, 5 * nb_sample))
    i = 0
    for file in os.listdir(gt_folder):
        pred_filename = f"seg_{best_model}_{file}"
        pred_path = os.path.join(pred_folder, pred_filename)

        gt_path = os.path.join(gt_folder, file)
        if not os.path.exists(pred_path):
            print(f"Prediction file not found: {pred_path}")
            continue

        gt_mask = Image.open(gt_path)
        pred_mask = Image.open(pred_path)

        idx = np.random.randint(0, nb_sample)

        axes[i, 0].imshow(gt_mask, cmap='gray')
        axes[i, 0].set_title(f"GT - {file}", fontsize=12)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pred_mask, cmap='gray')
        axes[i, 1].set_title(f"Pred - {file}", fontsize=12)
        axes[i, 1].axis('off')

        i += 1
        if i >= nb_sample:
            break

    plt.tight_layout()
    plt.savefig("plots/val_predictions_comparison.png")
    plt.show()


def plot_pred_sample(images_folder, pred_folder, best_model="33"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    if not image_files:
        print(f"No images found in {images_folder}")
        return

    random_sample = np.random.choice(image_files)
    filename = random_sample.split(".")[0]
    image_path = os.path.join(images_folder, random_sample)

    pred_filename = f"seg_{best_model}_{filename}.tif"
    pred_path = os.path.join(pred_folder, pred_filename)

    if not os.path.exists(pred_path):
        print(f"Prediction file not found: {pred_path}")
        print("Make sure to run the prediction script to generate the segmentation masks before visualizing.")
        return

    image = plt.imread(image_path)
    pred_mask = Image.open(pred_path)

    axes[0].imshow(image)
    axes[0].set_title(f"Image - {random_sample}", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title(f"Pred - {random_sample}", fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(image)
    axes[2].imshow(pred_mask, cmap='gray', alpha=0.7)
    axes[2].set_title(f"Overlay - {random_sample}", fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f"plots/sample_prediction_{filename}.png")
    plt.show()


if __name__ == "__main__":

    plot_sample(f"train_patches_{cfg.PATCH_SIZE}")

    # pred_folder = os.path.join(cfg.MAIN_DATA_PATH, "test", "predictions")
    # compare_predictions("bellingham10", pred_folder, ["1", "14", "33"])

    # gt_folder = os.path.join(cfg.MAIN_DATA_PATH, "val", "gt")
    # pred_folder = os.path.join(cfg.MAIN_DATA_PATH, "val", "predictions")
    # plot_val_pred(gt_folder, pred_folder, nb_sample=3, best_model="33")

    # images_folder = os.path.join(cfg.MAIN_DATA_PATH, "test", "images")
    # pred_folder = os.path.join(cfg.MAIN_DATA_PATH, "test", "predictions")
    # plot_pred_sample(images_folder, pred_folder, best_model="33")

