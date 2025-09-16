import torch.nn as nn
from torchvision import models

def Resnet_50(output_len):

    model = models.resnet50(pretrained=True)

    model.fc = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(512, output_len)
    )

    return model


