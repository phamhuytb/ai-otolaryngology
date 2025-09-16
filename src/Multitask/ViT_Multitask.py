from utils.data_preprocessing import prepare_dataset_multitask
from utils.trainers.models import ViT_Multitask
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from utils.evaluation.validate_model_multitask import compute_accuracy, compute_f1_score, compute_precision_score, compute_recall_score
from utils.data_visualization import Visualize_Metric_Multitask
import torch
import yaml
import os
import json
import time

CONFIG_PATH = "../../config"
config_name = "config_dataset_multitask.yaml"
# Load config
with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

# Load dataset
train_dataloaders, val_dataloaders, annotations = prepare_dataset_multitask.load_datasets(config)

# Load model
model = ViT_Multitask(config, annotations)

# Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = config['epochs']
learning_rate = config['learning_rate']
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5, factor = 0.8)

# Metric
metric = {
        'train': {
                'loss': [],
                'loss ent': [],
                'loss diseases': [],
                'accuracy': [],
                'f1 score': [],
                'precision score': [],
                'recall score': [],
        },
        'val': {
                'loss': [],
                'loss ent': [],
                'loss diseases': [],
                'accuracy': [],
                'f1 score': [],
                'precision score': [],
                'recall score': [],
        },
        'epochs': epochs,
        'learning rate': [],
        'time': None
}

# Training
model.to(DEVICE)
start_time_train = time.time()
for epoch in range(epochs):
        print('\n\nSTART EPOCH {}: '.format(epoch), end='')

        # Training
        loss_batch = 0.0
        loss_ent = 0.0
        loss_diseases = 0.0
        acc_batch = 0.0
        f1_batch = 0.0
        precision_batch = 0.0
        recall_batch = 0.0
        model.train()
        for idx, batch in enumerate(train_dataloaders):
                print('-', end='')

                images = batch[0]
                images = images.to(DEVICE)

                task_ent, task_disease = batch[1]
                task_ent = task_ent.to(DEVICE)
                task_disease = task_disease.to(DEVICE)

                # zero parameter gradient
                optimizer.zero_grad()

                # Forward
                train_output1, train_output2 = model(images)

                # Compute individual losses
                loss1 = criterion(train_output1, task_ent)
                loss2 = criterion(train_output2, task_disease)
                loss_ent += loss1.item()
                loss_diseases += loss2.item()

                # accuracy
                true_task = task_ent.detach().cpu().numpy()
                true_diesease = task_disease.detach().cpu().numpy()
                out1 = train_output1.detach().cpu().numpy()
                out2 = train_output2.detach().cpu().numpy()
                # compute
                avg_accuracy, acc1, acc2 = compute_accuracy((true_task, true_diesease), (out1, out2 ))
                avg_f1, f1_1, f1_2 = compute_f1_score((true_task, true_diesease), (out1, out2 ))
                avg_precision, precision1, precision2 = compute_precision_score((true_task, true_diesease), (out1, out2 ))
                avg_recall, recall1, recall2 = compute_recall_score((true_task, true_diesease), (out1, out2 ))

                acc_batch += avg_accuracy
                f1_batch += avg_f1
                precision_batch += avg_precision
                recall_batch += avg_recall

                # Combine the losses
                combined_loss = loss1 + loss2
                loss_batch += combined_loss.item()

                # Example training step
                combined_loss.backward()
                optimizer.step()
        # Loss - Accuracy
        loss_avg = loss_batch/len(train_dataloaders)
        loss_ent_avg = loss_ent/len(train_dataloaders)
        loss_diseases_avg = loss_diseases/len(train_dataloaders)
        acc_avg = acc_batch/len(train_dataloaders)
        f1_avg = f1_batch/len(train_dataloaders)
        precision_avg = precision_batch/len(train_dataloaders)
        recall_avg = recall_batch/len(train_dataloaders)
        # Add to metric
        metric['train']['loss'].append(loss_avg)
        metric['train']['loss ent'].append(loss_ent_avg)
        metric['train']['loss diseases'].append(loss_diseases_avg)
        metric['train']['accuracy'].append(acc_avg)
        metric['train']['f1 score'].append(f1_avg)
        metric['train']['precision score'].append(precision_avg)
        metric['train']['recall score'].append(recall_avg)
        # View
        print('\n\tTRAIN')
        print('\t\tLoss =', loss_avg)
        print('\t\tAccuracy = {:.2f} %'.format(acc_avg * 100))
        print('\t\tF1 Score = {:.2f} %'.format(f1_avg * 100))
        print('\t\tPrecision Score = {:.2f} %'.format(precision_avg * 100))
        print('\t\tRecall Score = {:.2f} %'.format(recall_avg * 100))

        # ==================================================================================================
        # Validation
        loss_batch = 0.0
        loss_ent = 0.0
        loss_diseases = 0.0
        acc_batch = 0.0
        f1_batch = 0.0
        precision_batch = 0.0
        recall_batch = 0.0
        model.eval()
        with torch.no_grad():
                for idx, batch in enumerate(val_dataloaders):
                        images = batch[0]
                        images = images.to(DEVICE)

                        task_ent, task_disease = batch[1]
                        task_ent = task_ent.to(DEVICE)
                        task_disease = task_disease.to(DEVICE)

                        # Forward
                        val_output1, val_output2 = model(images)

                        # Compute individual losses
                        loss1 = criterion(val_output1, task_ent)
                        loss2 = criterion(val_output2, task_disease)
                        loss_ent += loss1.item()
                        loss_diseases += loss2.item()

                        # accuracy
                        true_task = task_ent.detach().cpu().numpy()
                        true_diesease = task_disease.detach().cpu().numpy()
                        out1 = val_output1.detach().cpu().numpy()
                        out2 = val_output2.detach().cpu().numpy()

                        avg_accuracy, acc1, acc2 = compute_accuracy((true_task, true_diesease), (out1, out2 ))
                        avg_f1, f1_1, f1_2 = compute_f1_score((true_task, true_diesease), (out1, out2 ))
                        avg_precision, precision1, precision2 = compute_precision_score((true_task, true_diesease), (out1, out2 ))
                        avg_recall, recall1, recall2 = compute_recall_score((true_task, true_diesease), (out1, out2 ))

                        acc_batch += avg_accuracy
                        f1_batch += avg_f1
                        precision_batch += avg_precision
                        recall_batch += avg_recall

                        # Combine the losses
                        combined_loss = loss1 + loss2
                        loss_batch += combined_loss.item()
        # Loss - Accuracy
        loss_avg = loss_batch/len(val_dataloaders)
        loss_ent_avg = loss_ent/len(val_dataloaders)
        loss_diseases_avg = loss_diseases/len(val_dataloaders)
        acc_avg = acc_batch/len(val_dataloaders)
        f1_avg = f1_batch/len(val_dataloaders)
        precision_avg = precision_batch/len(val_dataloaders)
        recall_avg = recall_batch/len(val_dataloaders)
        # Add to metric
        metric['val']['loss'].append(loss_avg)
        metric['val']['loss ent'].append(loss_ent_avg)
        metric['val']['loss diseases'].append(loss_diseases_avg)
        metric['val']['accuracy'].append(acc_avg)
        metric['val']['f1 score'].append(f1_avg)
        metric['val']['precision score'].append(precision_avg)
        metric['val']['recall score'].append(recall_avg)
        # View
        print('\tVAL')
        print('\t\tLoss =', loss_avg)
        print('\t\tAccuracy = {:.2f} %'.format(acc_avg*100))
        print('\t\tF1 Score = {:.2f} %'.format(f1_avg * 100))
        print('\t\tPrecision Score = {:.2f} %'.format(precision_avg * 100))
        print('\t\tRecall Score = {:.2f} %'.format(recall_avg * 100))

        # =============================================================================
        print("\tLearning Rate = {}".format(optimizer.param_groups[0]['lr']))
        metric['learning rate'].append(optimizer.param_groups[0]['lr'])


# END Time
end_time_train = time.time()
toltal_time = end_time_train - start_time_train
hours, rem = divmod(toltal_time, 3600)
minutes, seconds = divmod(rem, 60)
metric['time'] = toltal_time
print('\n\nEND TRAINING')
print(f'\tToltal Time: {int(hours)}h {int(minutes)}m {int(seconds)}s')

# SAVE MODEL
model_path = config['output_path'] + '/model.pth'
torch.save(model.state_dict(), model_path)

# SAVE IMAGE
image_path = config['output_path'] + '/visualize.png'
Visualize_Metric_Multitask.plot_metric(metric, save2image=True, path=image_path)

# SAVE METRIC
metric_path = config['output_path'] + '/metric.json'
with open(metric_path, 'w') as json_file:
        json.dump(metric, json_file, indent=4)

# SAVE ANNOTATION
metric_path = config['output_path'] + '/annotation.json'
with open(metric_path, 'w') as json_file:
        del annotations['train']
        del annotations['val']
        json.dump(annotations, json_file, indent=4)