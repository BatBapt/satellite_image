import torchvision
import torch.nn as nn
from torchsummary import summary

import configuration as cfg


class SatelliteDeepLab(nn.Module):
    def __init__(self, num_classes):
        super(SatelliteDeepLab, self).__init__()

        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        )

        self.model.aux_classifier = None

        self.model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)

    def forward(self, x):
        return self.model(x)["out"]


if __name__ == "__main__":

    model = SatelliteDeepLab(num_classes=1).to(cfg.DEVICE)
    print(model)