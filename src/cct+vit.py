
from pathlib import Path
import time
import torch
from sympy.physics.units import momentum
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn

# Load additional utilities and functions
from utils.load_config import load_config
from utils.data_preprocessing.prepare_dataset_single_classifier import load_datasets
from utils.trainers.models import *
from utils.trainers import train_one_epoch
from utils.evaluation import save_metrics_plots, validate_one_epoch
from utils.trainers import EarlyStopper
#
# Load the configuration file
config = load_config(r"../config/all_in.yaml")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
train_loader, val_loader, train_dataset_classes = load_datasets(config)
id2label = {idx: label for idx, label in enumerate(train_dataset_classes)}
# label2id = {label: idx for idx, label in enumerate(train_dataset_classes)}

# Initialize the model and set up configurations
model1 = cct_14_7x2_224(pretrained=True, progress=True,num_classes=len(train_dataset_classes))
model1.fc = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(in_features=384, out_features=124, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(in_features=124, out_features=len(train_dataset_classes),bias=True)
    )

model2 = Vit(num_classes=len(train_dataset_classes))
model = MergedModel(model1,model2)

model.to(device)
print(id2label)

# Initialize optimizer and loss criterion
optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'],betas =(0.9, 0.999),eps =1e-8,weight_decay=0.1)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min' ,factor=0.9, patience=3, verbose=True)

# Setup directories for saving model and metrics
model_path = Path(config['training']['model_output'])
model_path.mkdir(parents=True, exist_ok=True)

# Metrics and training parameters initialization
metrics = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "train_f1": [],
    "train_precision": [],
    "train_sensitive": [],
    "train_specificity": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_f1": [],
    "val_precision": [],
    "val_sensitive": [],
    "val_specificity": [],

    "learning_rate": [],
    "train_time": []
}

total_epochs = config['training']['num_epochs']
best_f1_score = 0
epoch_train_times = []

# Calculate total number of trainable parameters
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
early_stopper = EarlyStopper(patience=5, min_delta=0.09)

def l1_penalty(var):
    return torch.sum(torch.abs(var))
# Training loop
for epoch in range(total_epochs):
    train_loss, train_accuracy, train_f1_score,train_precision,train_sensitive,train_specificity, train_duration = train_one_epoch(model, train_loader, optimizer,
                                                                                 criterion, device)
    val_loss, val_accuracy, val_f1_score,val_precision,val_sensitive,val_specificity = validate_one_epoch(model, val_loader, criterion, device)
    # l1_loss = sum(l1_penalty(param) for param in model.parameters())

    if early_stopper.early_stop(val_loss):
        break
    epoch_train_times.append(train_duration)

    # Update metrics and learning rate
    current_lr = optimizer.param_groups[0]['lr']
    metrics["learning_rate"].append(current_lr)
    metrics["train_time"].append(train_duration)

    metrics["epoch"].append(epoch + 1)
    metrics["train_loss"].append(train_loss)
    metrics["train_accuracy"].append(train_accuracy)
    metrics["train_f1"].append(train_f1_score)
    metrics["train_precision"].append(train_precision)
    metrics["train_sensitive"].append(train_sensitive)
    metrics["train_specificity"].append(train_specificity)
    metrics["val_loss"].append(val_loss)
    metrics["val_accuracy"].append(val_accuracy)
    metrics["val_f1"].append(val_f1_score)
    metrics["val_precision"].append(val_precision)
    metrics["val_sensitive"].append(val_sensitive)
    metrics["val_specificity"].append(val_specificity)

    scheduler.step(val_loss)  # Update learning rate based on the validation loss

    # Save the best model
    if val_f1_score > best_f1_score:
        best_f1_score = val_f1_score
        best_model_path = model_path
        # model.save_pretrained(best_model_path, from_pt=True)
        torch.save(model.state_dict(), str(best_model_path) + '/weights.h5')

    print(
        f"Epoch {epoch + 1}/{total_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1_score:.4f}, Train Precision: {train_precision:.4f},Train Sensitive: {train_sensitive:.4f},Train Specificity: {train_specificity:.4f}")
    print(
        f"Epoch {epoch + 1}/{total_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1_score:.4f}, Val Precision: {val_precision:.4f},Val Sensitive: {val_sensitive:.4f},Val Specificity: {val_specificity:.4f}")
    print(current_lr)
    total_train_time = sum(epoch_train_times)
    save_metrics_plots(metrics, config, total_params, total_train_time)


