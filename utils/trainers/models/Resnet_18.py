import torchvision.models as models
import torch.nn as nn


def resnet_18(len):
    model = models.resnet18(pretrained=True)

    model.fc = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=256, out_features=len, bias=True)

    )


    return model
