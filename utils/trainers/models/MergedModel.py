import torch
import torch.nn as nn


class MergedModel(nn.Module):
    def __init__(self, model1, model2,num_classes):
        super(MergedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        )


    def forward(self, x):
        # Get the outputs from both models
        output1 = self.model1(x)
        output2 = self.model2(x)


        combined_output = torch.cat((output1,output2),dim=1)
        final_output= self.classifier(combined_output)


        return final_output
