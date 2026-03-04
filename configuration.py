import torch
import os
import yaml


# Torch settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_MODEL_PATH = "D:/models/torch/hub"  # Update path for your need, this is where pytorch models will be downloaded and stored
torch.hub.set_dir(TORCH_MODEL_PATH) # Un/comment this line to un/set the previous directory


# Load config
config_path = os.path.join(os.curdir, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Paths
MAIN_DATA_PATH = config["settings"]["data_path"]
MODEL_WEIGHTS_PATH = os.path.join(os.curdir, "models")


# Hyperparameters & Global settings
PATCH_SIZE = config["settings"]["patch_size"]
BATCH_SIZE = config["hyperparameters"]["batch_size"]
EPOCHS = config["hyperparameters"]["epochs"]
LR_BACKBONE = config["hyperparameters"]["lr_backbone"]
LR_CLASSIFIER = config["hyperparameters"]["lr_classifier"]


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Main data path: {MAIN_DATA_PATH}")
    print(f"Model weights path: {MODEL_WEIGHTS_PATH}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate for backbone: {LR_BACKBONE}")
    print(f"Learning rate for classifier: {LR_CLASSIFIER}")
