from torchvision import models
import torch.nn as nn
import torch


class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.model = models.efficientnet_b3(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(in_features=1536, out_features=768, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(in_features=768, out_features=num_classes, bias=True)
        )


        freeze_start = 'features.1.0.block.0.0.weight'
        freeze_end = 'features.3.0.block.0.0.weight'
        freeze_flag = False

        for name, param in self.model.named_parameters():
            if name == freeze_start:
                freeze_flag = True
            if name == freeze_end:
                freeze_flag = False
            param.requires_grad = not freeze_flag

        # self.model = models.efficientnet_b1(pretrained=True)
        # self.model.classifier =  torch.nn.Identity()
        # self.model.classifier = nn.Sequential(
        #     nn.Dropout(p=0.3, inplace=False),
        #     nn.Linear(in_features=1536, out_features=640, bias=True),
        #     nn.ReLU(inplace=False),
        #     nn.Dropout(p=0.3, inplace=False),
        #     nn.Linear(in_features=640, out_features=num_classes, bias=True)
        # )
        #
        # freeze_start = 'features.1.0.block.0.0.weight'
        # freeze_end = 'features.3.0.block.0.0.weight'
        # freeze_flag = False
        #
        # for name, param in self.model.named_parameters():
        #     if name == freeze_start:
        #         freeze_flag = True
        #     if name == freeze_end:
        #         freeze_flag = False
        #     param.requires_grad = not freeze_flag

    def forward(self, x):
        return self.model(x)



