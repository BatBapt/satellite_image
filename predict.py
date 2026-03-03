import os
import torch
from torchvision import tv_tensors
import math
from PIL import Image

from models import SatelliteDeepLab
from dataset_utils import get_transform
import configuration as cfg


def pred(image_path, model, device, output_path, patch_size=256, batch_size=16):
    print(f"Processing image: {image_path}")

    model.eval()

    img = Image.open(image_path).convert("RGB")
    original_w, original_h = img.size

    new_w = math.ceil(original_w / patch_size) * patch_size
    new_h = math.ceil(original_h / patch_size) * patch_size

    padded_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded_img.paste(img, (0, 0))

    patches_tensor = []
    coords = []

    for y in range(0, new_h, patch_size):
        for x in range(0, new_w, patch_size):
            patch = padded_img.crop((x, y, x + patch_size, y + patch_size))

            tensor_patch = get_transform(train=False)(patch)

            patches_tensor.append(tensor_patch)
            coords.append((x, y))

    print(f"\tImage divided into {len(patches_tensor)} patches of size {patch_size}x{patch_size}")


    full_mask = torch.zeros((new_h, new_w), dtype=torch.float32)

    with torch.no_grad():
        for i in range(0, len(patches_tensor), batch_size):
            batch_images = torch.stack(patches_tensor[i: i + batch_size]).to(device)

            outputs = model(batch_images)


            probs = torch.sigmoid(outputs)

            probs = probs.squeeze(1).cpu()

            for j, (x, y) in enumerate(coords[i: i + batch_size]):
                full_mask[y: y + patch_size, x: x + patch_size] = probs[j]

    final_mask = full_mask[:original_h, :original_w]

    final_mask_bin = (final_mask > 0.5).to(torch.uint8) * 255

    result_img = Image.fromarray(final_mask_bin.numpy(), mode="L")
    result_img.save(output_path)

    print(f"\tSegmentation mask generated and saved to {output_path}")


if __name__ == "__main__":

    model = SatelliteDeepLab(num_classes=1)

    best_weight = os.path.join(cfg.MODEL_WEIGHTS_PATH, "satellite_deeplab", "best_model_1.pth")
    suffix_model = best_weight.split(os.sep)[-1].split(".")[0].split("_")[-1]  # 'best_model_XXX.pth' -> 'XXX'

    model.load_state_dict(torch.load(best_weight, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)

    test_image_path = os.path.join(cfg.MAIN_DATA_PATH, "test", "images")
    output_seg_paths = os.path.join(cfg.MAIN_DATA_PATH, "test", "predictions")
    os.makedirs(output_seg_paths, exist_ok=True)

    for file in os.listdir(test_image_path):
        filename, ext = file.split(".")
        image_path = os.path.join(test_image_path, file)
        output_path = os.path.join(output_seg_paths, f"seg_{suffix_model}_{filename}.{ext}")
        pred(image_path, model, cfg.DEVICE, output_path, patch_size=256, batch_size=cfg.BATCH_SIZE)