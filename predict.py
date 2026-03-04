import os
import math
import torch
from PIL import Image
from tqdm import tqdm  # Recommended for progress bars

from models import SatelliteDeepLab
from dataset_utils import get_transform
import configuration as cfg


def get_patch_generator(image, patch_size, transform):
    """
    Yields patches and their coordinates to save memory.
    """
    w, h = image.size
    # Calculate padding
    new_w = math.ceil(w / patch_size) * patch_size
    new_h = math.ceil(h / patch_size) * patch_size

    padded_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded_img.paste(image, (0, 0))

    for y in range(0, new_h, patch_size):
        for x in range(0, new_w, patch_size):
            patch = padded_img.crop((x, y, x + patch_size, y + patch_size))
            tensor_patch = transform(patch)
            yield tensor_patch, (x, y)


def pred(image_path, model, device, output_path, patch_size=256, batch_size=16, verbose=True):
    if verbose:
        print(f"Processing image: {image_path}")

    model.eval()

    # Load Image
    img = Image.open(image_path).convert("RGB")
    original_w, original_h = img.size

    # Calculate new dimensions for the mask buffer
    new_w = math.ceil(original_w / patch_size) * patch_size
    new_h = math.ceil(original_h / patch_size) * patch_size

    # Initialize full mask on CPU to hold results
    full_mask = torch.zeros((new_h, new_w), dtype=torch.float32)

    # Prepare transform once
    transform = get_transform(train=False)

    # Create generator
    patch_gen = get_patch_generator(img, patch_size, transform)

    batch_images = []
    batch_coords = []

    with torch.no_grad():
        # Iterate through the generator
        for tensor_patch, (x, y) in patch_gen:
            batch_images.append(tensor_patch)
            batch_coords.append((x, y))

            # When batch is full, run inference
            if len(batch_images) == batch_size:
                input_tensor = torch.stack(batch_images).to(device)
                outputs = model(input_tensor)
                probs = torch.sigmoid(outputs).squeeze(1).cpu()

                # Place result into full mask
                for k, (bx, by) in enumerate(batch_coords):
                    full_mask[by: by + patch_size, bx: bx + patch_size] = probs[k]

                # Reset batch
                batch_images = []
                batch_coords = []

        # Process remaining patches (last incomplete batch)
        if batch_images:
            input_tensor = torch.stack(batch_images).to(device)
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze(1).cpu()

            for k, (bx, by) in enumerate(batch_coords):
                full_mask[by: by + patch_size, bx: bx + patch_size] = probs[k]

    # Crop back to original size
    final_mask = full_mask[:original_h, :original_w]

    # Threshold and convert to image
    final_mask_bin = (final_mask > 0.5).to(torch.uint8) * 255
    result_img = Image.fromarray(final_mask_bin.numpy(), mode="L")
    result_img.save(output_path)

    if verbose:
        print(f"\tSegmentation mask generated and saved to {output_path}")


if __name__ == "__main__":
    # Setup paths
    test_image_path = os.path.join(cfg.MAIN_DATA_PATH, "val", "images")
    output_seg_paths = os.path.join(cfg.MAIN_DATA_PATH, "val", "predictions")
    os.makedirs(output_seg_paths, exist_ok=True)

    # Load Model
    model = SatelliteDeepLab(num_classes=1)
    best_weight = os.path.join(cfg.MODEL_WEIGHTS_PATH, f"satellite_deeplab_{cfg.PATCH_SIZE}", "best_model_32.pth")

    suffix_model = os.path.splitext(os.path.basename(best_weight))[0].split("_")[-1]

    print(f"Loading weights from: {best_weight}")
    model.load_state_dict(torch.load(best_weight, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)

    all_files = [f for f in os.listdir(test_image_path) if
                 f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    total_files = len(all_files)

    print(f"Found {total_files} images to process.")

    # Loop
    for i, file in enumerate(all_files):
        print(f"Processing file {i + 1}/{total_files}: {file}")

        # Robust extension handling
        filename, ext = os.path.splitext(file)

        image_path = os.path.join(test_image_path, file)
        # Construct output filename: seg_XXX_filename.ext
        output_filename = f"seg_{suffix_model}_{filename}{ext}"
        output_path = os.path.join(output_seg_paths, output_filename)

        if os.path.exists(output_path):
            continue

        pred(
            image_path,
            model,
            cfg.DEVICE,
            output_path,
            patch_size=cfg.PATCH_SIZE,
            batch_size=cfg.BATCH_SIZE,
            verbose=False
        )