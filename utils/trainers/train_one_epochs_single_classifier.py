import torch
from tqdm import tqdm
from utils.evaluation.metric import compute_accuracy, compute_f1_score,compute_precision,compute_sensitive,compute_specificity
import numpy as np
import time
l1_lambda = 0.000003
l2_lambda = 0.00001

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    start_time = time.time()
    train_loss = []
    true_labels = []
    pred_labels = []
    total_loss = 0
    for inputs, labels in tqdm(train_loader, desc='Training'):
        temp_loss=0

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # l1 = sum(p.abs().sum() for p in model.parameters())

        # l2 = sum(p.pow(2.0).sum() for p in model.parameters())

        loss = criterion(outputs, labels)
        # loss+=  l2*l2_lambda
        temp_loss += loss.item()
        train_loss.append(temp_loss/len(train_loader))

        loss.backward()
        optimizer.step()

        # train_loss.append(loss_batch)
        _, predictions = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predictions.cpu().numpy())
    end_time = time.time()
    train_duration = end_time - start_time

    train_accuracy = compute_accuracy(true_labels, pred_labels)
    train_f1_score = compute_f1_score(true_labels, pred_labels, average='macro')
    train_precision= compute_precision(true_labels, pred_labels, average='macro')
    train_sensitive = compute_sensitive(true_labels, pred_labels, average='macro')
    train_specificity = compute_specificity(true_labels, pred_labels, average='macro')
    return np.mean(train_loss), train_accuracy, train_f1_score,train_precision,train_sensitive,train_specificity,train_duration
