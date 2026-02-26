import os.path

import matplotlib.pyplot as plt
import numpy as np
import configuration as paths

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


if __name__ == "__main__":

    plot_sample("train_patches")

