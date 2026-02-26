import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors
import shutil

import configuration as paths


def get_transform(train):
    if train:
        transform_list = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomCrop((224, 224)),
            v2.ToImage(),  # Remplace ToTensor()
            v2.ToDtype(torch.float32, scale=True), # Convertit en float [0, 1]
            # La normalisation ne s'appliquera automatiquement qu'à l'image
            v2.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
        ])
    else:
        transform_list = v2.Compose([
            v2.CenterCrop((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
        ])

    return transform_list


def format_dataset(input_path, split_ratio=0.2):
    """
    Function used to create a val dataset from the train dataset with a random split using the ratio parameter
    Args:
        input_path:
    Returns:
    """
    train_path = os.path.join(input_path, "train")
    val_path = os.path.join(input_path, "val")

    os.makedirs(val_path, exist_ok=True)

    train_images = os.path.join(train_path, "images")
    train_labels = os.path.join(train_path, "gt")

    val_images = os.path.join(val_path, "images")
    val_labels = os.path.join(val_path, "gt")

    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)

    if os.path.exists(val_images) and os.path.exists(val_labels):
        return

    all_images = sorted(os.listdir(train_images))
    num_val_samples = int(len(all_images) * split_ratio)
    val_samples = set(random.sample(all_images, num_val_samples))

    for img_name in all_images:
        if img_name in val_samples:
            shutil.move(os.path.join(train_images, img_name), os.path.join(val_images, img_name))

    for label_name in all_images:
        if label_name in val_samples:
            shutil.move(os.path.join(train_labels, label_name), os.path.join(val_labels, label_name))


def create_patches(input_dir, output_dir, patch_size=256):
    out_images_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "gt")

    if os.path.exists(out_images_dir) and os.path.exists(out_labels_dir):
        return

    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "gt")

    for filename in os.listdir(images_dir):
        img_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, filename)

        try:
            img = Image.open(img_path).convert("RGB")
            label = Image.open(label_path).convert("L")
        except Exception as e:
            print(f"Error with {filename}: {e}")
            continue

        w, h = img.size

        patch_idx = 0
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                box = (x, y, x + patch_size, y + patch_size)

                patch_img = img.crop(box)
                patch_label = label.crop(box)

                base_name, ext = os.path.splitext(filename)
                new_name = f"{base_name}_patch_{patch_idx}{ext}"

                patch_img.save(os.path.join(out_images_dir, new_name))
                patch_label.save(os.path.join(out_labels_dir, new_name))

                patch_idx += 1
    print(f"End of the process, {patch_idx} patches created for each image.")


class CustomDataset(Dataset):
    def __init__(self, input_path, transforms=None):
        self.input_path = input_path

        self.images_files = os.path.join(self.input_path, "images")
        self.labels_files = os.path.join(self.input_path, "gt")

        self.all_images = sorted(os.listdir(self.images_files))
        self.all_labels = sorted(os.listdir(self.labels_files))
        self.transforms = transforms

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        label_name = self.all_labels[idx]

        image_path = os.path.join(self.images_files, image_name)
        label_path = os.path.join(self.labels_files, label_name)

        image = Image.open(image_path)
        label = Image.open(label_path)

        # 1. On indique explicitement à PyTorch la nature de chaque donnée
        image = tv_tensors.Image(image)
        label = tv_tensors.Mask(label) # Crucial: v2 saura qu'il ne faut pas normaliser ceci !

        # 2. On passe les DEUX variables en même temps
        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label


if __name__ == "__main__":
    format_dataset(paths.MAIN_DATA_PATH, split_ratio=0.2)

    train_path = os.path.join(paths.MAIN_DATA_PATH, "train")
    val_path = os.path.join(paths.MAIN_DATA_PATH, "val")

    train_patches_dir = os.path.join(paths.MAIN_DATA_PATH, "train_patches")
    val_patches_dir = os.path.join(paths.MAIN_DATA_PATH, "val_patches")

    create_patches(train_path, train_patches_dir, patch_size=256)
    create_patches(val_path, val_patches_dir, patch_size=256)

    dataset = CustomDataset(input_path=train_patches_dir)
    print(f"Dataset size: {len(dataset)}")
    sample_image, sample_label = dataset[0]
    print(f"Sample image shape: {sample_image.size()}")
    print(f"Sample label shape: {sample_label.size()}")