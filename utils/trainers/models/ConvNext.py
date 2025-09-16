from torch import nn
from transformers import ConvNextV2Model


class CustomConvNextV2Model(ConvNextV2Model):
    def __init__(self, config,num_class):
        super().__init__(config)
        # self.classifier = nn.Linear(in_features=640, out_features=num_class, bias=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=640, out_features=320, bias=True),
            nn.GELU(),

            nn.BatchNorm1d(320),  # Adding BatchNorm1d
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=320, out_features=num_class, bias=True)
        )
    def forward(self, pixel_values):
        # Get the outputs from the base model
        outputs = super().forward(pixel_values)

        pooled_output = outputs.pooler_output

        logits = self.classifier(pooled_output)

        return logits