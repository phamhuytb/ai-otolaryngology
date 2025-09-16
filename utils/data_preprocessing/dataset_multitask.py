import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# Custom dataset for multitask learning - HuggingFace
# class CustomImageDataset(Dataset):
#         def __init__(self, annotations, processor):
#                 self.annotations = annotations
#                 self.processor = processor
#
#         def __len__(self):
#                 return len(self.annotations['images'])
#
#         def __getitem__(self, idx):
#                 # image
#                 path_img = self.annotations['images'][idx]
#                 image = Image.open(path_img).convert('RGB')
#                 image_ts = self.processor(image, return_tensors='pt')
#
#                 # label
#                 label = self.annotations['labels'][idx]
#
#                 return image_ts['pixel_values'][0], label


# Custom dataset for multitask learning - transform
class CustomImageDataset(Dataset):
        def __init__(self, annotations, processor):
                self.annotations = annotations
                self.processor = processor

        def __len__(self):
                return len(self.annotations['images'])

        def __getitem__(self, idx):
                # image
                path_img = self.annotations['images'][idx]
                image = Image.open(path_img).convert('RGB')
                image_ts = self.processor(image)

                # label
                label = self.annotations['labels'][idx]

                return image_ts, label