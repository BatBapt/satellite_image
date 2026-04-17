import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_utils import CustomDataset, get_transform
from models import SatelliteDeepLab
import configuration as cfg


def eval_dataset(model, dataloader, device, threshold=0.5):
    model.eval()

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    print("Starting evaluation on the dataset...")

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)


            labels = (labels > 0).float()

            outputs = model(images)
            labels = labels.view_as(outputs)

            preds = torch.sigmoid(outputs)
            preds = (preds > threshold).float()

            preds_flat = preds.view(-1)
            labels_flat = labels.view(-1)

            total_tp += (preds_flat * labels_flat).sum().item()
            total_fp += (preds_flat * (1 - labels_flat)).sum().item()
            total_fn += ((1 - preds_flat) * labels_flat).sum().item()

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)

    return precision, recall, f1_score, iou


if __name__ == "__main__":
    val_patches_dir = os.path.join(cfg.MAIN_DATA_PATH, f"val_patches_{cfg.PATCH_SIZE}")
    val_dataset = CustomDataset(input_path=val_patches_dir, transforms=get_transform(train=False))
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    best_weight = os.path.join(cfg.MODEL_WEIGHTS_PATH, f"satellite_deeplab_{cfg.PATCH_SIZE}", "best_model_33.pth")
    model = SatelliteDeepLab(num_classes=1)
    model.load_state_dict(torch.load(best_weight, map_location=cfg.DEVICE))
    model.to(cfg.DEVICE)

    precision, recall, f1, iou = eval_dataset(model, val_loader, cfg.DEVICE)

    print("\n" + "=" * 40)
    print("FINAL EVALUATION REPORT")
    print("=" * 40)
    print(f"Precision : {precision * 100:.2f} %")
    print(f"Recall    : {recall * 100:.2f} %")
    print(f"F1 Score  : {f1 * 100:.2f} %")
    print(f"IoU       : {iou * 100:.2f} %")
    print("=" * 40)

    """           
    33 (best):
        ========================================
        FINAL EVALUATION REPORT
        ========================================
        Precision : 89.26 %
        Recall    : 88.31 %
        F1 Score  : 88.78 %
        IoU       : 79.83 %
        ========================================
    """