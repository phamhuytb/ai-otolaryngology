
<<<<<<< HEAD:src/ear/Mobilenet_ear.py
import os
import sys

import yaml

=======
>>>>>>> origin/dev:src/ear/Swin_ear.py
from pathlib import Path
import time
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn

# Load additional utilities and functions
from utils.load_config import load_config
from utils.data_preprocessing import load_datasets
<<<<<<< HEAD:src/ear/Mobilenet_ear.py
from utils.trainers.models import Mobinet
from utils.trainers import train_one_epoch
from utils.evaluation import save_metrics_plots, validate_one_epoch
=======
from utils.trainers.models import swin
from utils.trainers import train_one_epoch
from utils.evaluation import save_metrics_plots, validate_one_epoch
from utils.trainers import EarlyStopper
>>>>>>> origin/dev:src/ear/Swin_ear.py

# Load the configuration file
config = load_config(r"../../config/ear_classifier.yaml")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
train_loader, val_loader, train_dataset_classes = load_datasets(config)
<<<<<<< HEAD:src/ear/Mobilenet_ear.py
id2label = {idx: label for idx, label in enumerate(train_dataset_classes)}
label2id = {label: idx for idx, label in enumerate(train_dataset_classes)}

# Initialize the model and set up configurations
model = Mobinet()
model.config.id2label = id2label
model.config.label2id = label2id
model.to(device)

# Initialize optimizer and loss criterion
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min',min_lr=0.0001 ,factor=0.95, patience=5, verbose=True)
=======
# id2label = {idx: label for idx, label in enumerate(train_dataset_classes)}
# label2id = {label: idx for idx, label in enumerate(train_dataset_classes)}

# Initialize the model and set up configurations
model = swin(len(train_dataset_classes))
# model.config.id2label = id2label
# model.config.label2id = label2id
model.to(device)

# Initialize optimizer and loss criterion
optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'],weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min' ,factor=0.6, patience=5, verbose=True)
>>>>>>> origin/dev:src/ear/Swin_ear.py

# Setup directories for saving model and metrics
model_path = Path(config['training']['model_output'])
model_path.mkdir(parents=True, exist_ok=True)

# Metrics and training parameters initialization
metrics = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "train_f1": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_f1": [],
    "learning_rate": [],
    "train_time": []
}

total_epochs = config['training']['num_epochs']
best_f1_score = 0
epoch_train_times = []

# Calculate total number of trainable parameters
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
<<<<<<< HEAD:src/ear/Mobilenet_ear.py

=======
early_stopper = EarlyStopper(patience=5, min_delta=0.05)

def l1_penalty(var):
    return torch.sum(torch.abs(var))
>>>>>>> origin/dev:src/ear/Swin_ear.py
# Training loop
for epoch in range(total_epochs):
    train_loss, train_accuracy, train_f1_score, train_duration = train_one_epoch(model, train_loader, optimizer,
                                                                                 criterion, device)
    val_loss, val_accuracy, val_f1_score = validate_one_epoch(model, val_loader, criterion, device)
<<<<<<< HEAD:src/ear/Mobilenet_ear.py
=======
    l1_loss = sum(l1_penalty(param) for param in model.parameters())

    if early_stopper.early_stop(val_loss):
        break
>>>>>>> origin/dev:src/ear/Swin_ear.py
    epoch_train_times.append(train_duration)

    # Update metrics and learning rate
    current_lr = optimizer.param_groups[0]['lr']
    metrics["learning_rate"].append(current_lr)
    metrics["train_time"].append(train_duration)

    metrics["epoch"].append(epoch + 1)
    metrics["train_loss"].append(train_loss)
    metrics["train_accuracy"].append(train_accuracy)
    metrics["train_f1"].append(train_f1_score)
    metrics["val_loss"].append(val_loss)
    metrics["val_accuracy"].append(val_accuracy)
    metrics["val_f1"].append(val_f1_score)

    scheduler.step(val_loss)  # Update learning rate based on the validation loss

    # Save the best model
    if val_f1_score > best_f1_score:
        best_f1_score = val_f1_score
        best_model_path = model_path
<<<<<<< HEAD:src/ear/Mobilenet_ear.py
        model.save_pretrained(best_model_path, from_pt=True)
=======
        # model.save_pretrained(best_model_path, from_pt=True)
        torch.save(model.state_dict(), str(best_model_path) + '/weights.h5')
>>>>>>> origin/dev:src/ear/Swin_ear.py

    print(
        f"Epoch {epoch + 1}/{total_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1_score:.4f}")
    print(
        f"Epoch {epoch + 1}/{total_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1_score:.4f}")
<<<<<<< HEAD:src/ear/Mobilenet_ear.py

# Calculate total training time
total_train_time = sum(epoch_train_times)

# Save training metrics and plots
save_metrics_plots(metrics, config, total_params, total_train_time)
=======
    print(current_lr)
    total_train_time = sum(epoch_train_times)
    save_metrics_plots(metrics, config, total_params, total_train_time)

>>>>>>> origin/dev:src/ear/Swin_ear.py
