import os
from torch.utils.data import DataLoader
import numpy as np
import torch
from transformers import AutoImageProcessor
from .dataset_multitask import CustomImageDataset
from .func_dataset_multitask import ResizeMin
from torchvision import transforms
from PIL import Image
import random


def get_annotations_dataset(root_path):
    annotations = {'num_tasks': None,
                   'num_diseases': None,
                   'list_tasks': None,
                   'list_diseases': None,
                   'train': {'images': [], 'labels': []},
                   'val': {'images': [], 'labels': []}}

    num_diseases = 0
    list_tasks = []
    list_diseases = []
    list_diseases_val = [] # copy để thứ tự train và val giống nhau

    # TRAIN
    count = 0
    path_mode = os.path.join(root_path, 'train')
    ent = os.listdir(path_mode)
    num_tasks = len(ent)
    # ENT
    for idx1, task in enumerate(ent):
            list_tasks.append(task)
            path_task = os.path.join(path_mode, task)
            list_disease = os.listdir(path_task)
            num_diseases += len(list_disease)
            list_diseases_val.append(list_disease)
            # Diseases
            for idx2, disease in enumerate(list_disease):
                    list_diseases.append(disease)
                    path_disease = os.path.join(path_task, disease)
                    list_images = os.listdir(path_disease)
                    # Image, Label
                    for idx3, image in enumerate(list_images):
                            path_images = os.path.join(path_disease, image)
                            annotations['train']['images'].append(path_images)
                            annotations['train']['labels'].append([idx1, count])
                    count += 1
    # VAL
    count = 0
    path_mode = os.path.join(root_path, 'val')
    # ENT
    for idx1, task in enumerate(ent):
            path_task = os.path.join(path_mode, task)
            # Diseases
            for idx2, disease in enumerate(list_diseases_val[idx1]):
                    path_disease = os.path.join(path_task, disease)
                    list_images = os.listdir(path_disease)
                    # Image, Label
                    for idx3, image in enumerate(list_images):
                            path_images = os.path.join(path_disease, image)
                            annotations['val']['images'].append(path_images)
                            annotations['val']['labels'].append([idx1, count])
                    count += 1

    annotations['num_tasks'] = num_tasks
    annotations['num_diseases'] = num_diseases
    annotations['list_tasks'] = list_tasks
    annotations['list_diseases'] = list_diseases
    return annotations


# Collate dataset
def my_collate(batch):
    images = [item[0].numpy() for item in batch]
    labels = [item[1] for item in batch]

    images_np = np.array(images, dtype=np.float32)
    labels_np = np.array(labels)

    images_ts = torch.tensor(images_np, dtype=torch.float32)
    labels_ts = (torch.tensor([i1[0] for i1 in labels_np]), torch.tensor([i2[1] for i2 in labels_np]))
    return images_ts, labels_ts

# Custom argument random rotation
class CustomRotate:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image):
        angle = random.uniform(-self.degrees, self.degrees)
        return image.rotate(angle, expand=True)


# Custom argument shift image
class CustomShift:
    def __init__(self, shift_range_x=(0.0, 0.3), shift_range_y=(0.0, 0.3)):
        self.shift_range_x = shift_range_x
        self.shift_range_y = shift_range_y

    def __call__(self, image):
        image_array = np.array(image)
        shift_x = int(image_array.shape[1] * np.random.uniform(*self.shift_range_x))
        shift_y = int(image_array.shape[0] * np.random.uniform(*self.shift_range_y))
        shifted_image_array = np.roll(image_array, shift=(shift_y, shift_x), axis=(0, 1))
        shifted_image = Image.fromarray(shifted_image_array)
        return shifted_image

# Custom random zoom and center crop
class RandomZoomCenterCrop:
    def __init__(self, zoom_range=(1.0, 1.2), size=(224, 224)):
        self.zoom_range = zoom_range
        self.size = size

    def __call__(self, image):
        width, height = image.size
        zoom_factor = random.uniform(*self.zoom_range)
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        image = image.resize((new_width, new_height), Image.BILINEAR)
        left = (new_width - self.size[0]) / 2
        top = (new_height - self.size[1]) / 2
        right = (new_width + self.size[0]) / 2
        bottom = (new_height + self.size[1]) / 2
        return image.crop((left, top, right, bottom))

def load_datasets(config):
     root_path = config['root_dataset']
     batch_size = config['batch_size']
     image_size = config['image_size']

     # Get annotations of datasets
     annotations = get_annotations_dataset(root_path)
     print('train:', len(annotations['train']['images']))
     print('val:', len(annotations['val']['images']))

     # Load preprocess image
     processor_train = transforms.Compose([
          ResizeMin(image_size + 4),
          transforms.CenterCrop(image_size+2),
          # 1
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomVerticalFlip(p=0.5),
          transforms.ColorJitter(brightness=(0.85, 1.3)),
          CustomShift(shift_range_x=(0.0, 0.5), shift_range_y=(0.0, 0.5)),
          CustomRotate(degrees=(45)),
          # 2
          transforms.Resize((image_size+2, image_size+2)),
          RandomZoomCenterCrop(zoom_range=(1.0, 1.15), size=(image_size, image_size)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
     ])

     processor_val = transforms.Compose([
        ResizeMin(image_size+4),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
     ])

     training_data = CustomImageDataset(annotations['train'], processor_train)
     valid_data = CustomImageDataset(annotations['val'], processor_val)

     train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=my_collate, shuffle=True)
     val_dataloader = DataLoader(valid_data, batch_size=batch_size, collate_fn=my_collate, shuffle=False)

     return train_dataloader, val_dataloader, annotations

# a = get_annotations_dataset('/home/bht/VKU/ThucTap_AI_Y_Te_PCT/data/Ear_Nose_Throat')
