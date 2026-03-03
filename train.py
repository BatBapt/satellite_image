import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SatelliteDeepLab
from dataset_utils import CustomDataset, get_transform
import configuration as cfg


def calculate_iou(preds, labels, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    labels = (labels > threshold).float()

    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection

    if union == 0:
        return torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)

    return intersection / union


def train_one_epoch(model, dataloader, criterion, optimizer, epoch, device):
    model.train()
    epoch_loss = 0.0
    epoch_iou = 0.0

    loop = tqdm(dataloader, leave=False, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        labels = (labels > 0).float()

        outputs = model(images)
        labels = labels.view_as(outputs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_iou += calculate_iou(outputs, labels).item()

        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(dataloader), epoch_iou / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            labels = (labels > 0).float()

            outputs = model(images)
            labels = labels.view_as(outputs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_iou += calculate_iou(outputs, labels).item()

    return val_loss / len(dataloader), val_iou / len(dataloader)


def main():
    if cfg.DEVICE.type == 'cuda':
        print(f"Training GPU on : {torch.cuda.get_device_name(0)}")
    else:
        print("/!\ Training on CPU")


    model_output_dir = os.path.join(cfg.MODEL_WEIGHTS_PATH, "satellite_deeplab")
    os.makedirs(model_output_dir, exist_ok=True)


    train_patches_dir = os.path.join(cfg.MAIN_DATA_PATH, "train_patches")
    val_patches_dir = os.path.join(cfg.MAIN_DATA_PATH, "val_patches")

    train_dataset = CustomDataset(input_path=train_patches_dir, transforms=get_transform(train=True))
    val_dataset = CustomDataset(input_path=val_patches_dir, transforms=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    print("Datasets and DataLoaders created !")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    model = SatelliteDeepLab(num_classes=1).to(cfg.DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam([
        {'params': model.model.backbone.parameters(), 'lr': cfg.LR_BACKBONE},
        {'params': model.model.classifier.parameters(), 'lr': cfg.LR_CLASSIFIER}
    ])

    best_iou = 0.0

    for epoch in range(cfg.EPOCHS):

        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, epoch, cfg.DEVICE)

        val_loss, val_iou = validate(model, val_loader, criterion, cfg.DEVICE)

        print(f"Train Loss: {train_loss:.4f}\tTrain IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}\tVal IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(model_output_dir, f"best_model_{epoch+1}.pth"))
            print("New best model saved!")

        if epoch + 1 == cfg.EPOCHS:
            torch.save(model.state_dict(), os.path.join(model_output_dir, f"final_model.pth"))

if __name__ == "__main__":
    main()