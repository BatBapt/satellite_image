import torch
import os

"""
LINKS: 
dataset_path : https://huggingface.co/datasets/blanchon/INRIA-Aerial-Image-Labeling
"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_MODEL_PATH = "D:/models/torch/hub"
torch.hub.set_dir(TORCH_MODEL_PATH) # Un/comment this line to un/set the directory to download pytorch models
MAIN_DATA_PATH = "D:/Programmation/IA/datas/AerialImageDataset"


MODEL_WEIGHTS_PATH = os.path.join(os.curdir, "models")
