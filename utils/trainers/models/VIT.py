import torchvision.models as models
import torch.nn as nn
import torch


class Vit(nn.Module):
    def __init__(self, num_classes):
        super(Vit, self).__init__()
        self.model = models.vit_b_16(pretrained=True)
        # self.model.heads= torch.nn.Identity()
        self.model.heads = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features=768, out_features=384, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features=384, out_features=num_classes, bias=True)
        )


        freeze_start = 'encoder.layers.encoder_layer_0.ln_1.weight'
        freeze_end = 'encoder.layers.encoder_layer_4.ln_1.weight'
        freeze_flag = False

        for name, param in self.model.named_parameters():
            if name == freeze_start:
                freeze_flag = True
            if name == freeze_end:
                freeze_flag = False
            param.requires_grad = not freeze_flag

    def forward(self, x):
        return self.model(x)

