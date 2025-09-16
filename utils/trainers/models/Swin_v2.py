import torch.nn as nn
from torchvision import models


def swin_v2(len):
    model = models.swin_v2_b(pretrained=True)

    model.head = nn.Sequential(
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features=1024, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features=512, out_features=len, bias=True)
    )
    freeze_start = 'features.1.0.norm1.weight'
    freeze_end = 'features.2.reduction.weight'

    freeze_start1 = 'features.3.0.norm1.weight'
    freeze_end1 = 'features.4.reduction.weight'

    freeze_group_1 = False
    freeze_group_2 = False

    for name, param in model.named_parameters():

        if name == freeze_start:
            freeze_group_1 = True
        elif name == freeze_end:
            freeze_group_1 = False

        if name == freeze_start1:
            freeze_group_2 = True
        elif name == freeze_end1:
            freeze_group_2 = False

        param.requires_grad = not (freeze_group_1 or freeze_group_2)
    return model
