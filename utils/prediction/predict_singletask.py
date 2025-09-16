from utils.trainers.models import Resnet_50,swin_v2,Vit,EfficientNet
import os
import yaml
import json
import torch
from PIL import Image
from torchvision import transforms
from utils.data_preprocessing.prepare_dataset_single_classifier import ResizeMin
import matplotlib.pyplot as plt

def predict_print(image_path, config):

    transform = transforms.Compose([
        ResizeMin(config['size']+2),
        transforms.CenterCrop(config['size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    model = Vit(len(config['id2label']))
    model.load_state_dict(torch.load(config['model_path'], map_location=torch.device('cpu')))

    img = Image.open(image_path)
    imgs = transform(img)
    img_tensor = imgs.unsqueeze(0)
    outputs = model(img_tensor)
    _, preds = torch.max(outputs, 1)
    predicted_label = preds.item()
    return config['id2label'][predicted_label]

# with open('/home/bht/pycharm/PCT/config/predict_singletask.yaml') as file:
#     config = yaml.safe_load(file)
# out = predict_print("/home/bht/VKU/ThucTap_AI_Y_Te_PCT/data/Ear_Nose_Throat/val/ear/viem_ong_tai_ngoai_cap/20.0304.000141_7_Images_25.jpg", config)
# print(out)


