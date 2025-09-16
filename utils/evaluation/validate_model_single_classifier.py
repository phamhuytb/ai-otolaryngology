import torch
from tqdm import tqdm
from utils.evaluation.metric import accuracy_score, compute_f1_score
import numpy as np

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    true_labels = []
    pred_labels = []


    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            temp_loss=0
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # logits = outputs.logits
            loss = criterion(outputs, labels)

            # val_loss.append(loss.item())
            temp_loss+= loss.item()
            val_loss.append(temp_loss/len(val_loader))
            _, predictions = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

    val_accuracy = accuracy_score(true_labels, pred_labels)
    val_f1_score = compute_f1_score(true_labels, pred_labels, average='macro')
    return np.mean(val_loss), val_accuracy, val_f1_score
