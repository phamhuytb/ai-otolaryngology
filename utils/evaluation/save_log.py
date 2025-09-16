import matplotlib.pyplot as plt
from pathlib import Path

def save_metrics_plots(metrics, config, total_params, total_train_time):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(metrics['epoch'], metrics['train_accuracy'], label='Train Accuracy')
    plt.plot(metrics['epoch'], metrics['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(metrics['epoch'], metrics['train_f1'], label='Train F1 Score')
    plt.plot(metrics['epoch'], metrics['val_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()

    metrics_path = Path(config['training']['log_outputs'])
    metrics_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(metrics_path / 'training_metrics_plot.png')
    plt.close()

    with open(metrics_path / 'training_metrics.txt', 'w') as f:
        f.write("Epoch Details:\n")
        for i in range(len(metrics['epoch'])):
            f.write(f"Epoch {metrics['epoch'][i]}:\n")
            f.write(f" Train Loss: {metrics['train_loss'][i]:.4f}\n")
            f.write(f" Train Accuracy: {metrics['train_accuracy'][i]:.4f}\n")
            f.write(f" Train F1 Score: {metrics['train_f1'][i]:.4f}\n")
            f.write(f" Train Precision: {metrics['train_precision'][i]:.4f}\n")
            f.write(f" Train Sensitive: {metrics['train_sensitive'][i]:.4f}\n")
            f.write(f" Train Specificity: {metrics['train_specificity'][i]:.4f}\n")

            f.write(f" Validation Loss: {metrics['val_loss'][i]:.4f}\n")
            f.write(f" Validation Accuracy: {metrics['val_accuracy'][i]:.4f}\n")
            f.write(f" Validation F1 Score: {metrics['val_f1'][i]:.4f}\n")
            f.write(f" Validation Precision: {metrics['val_precision'][i]:.4f}\n")
            f.write(f" Validation Sensitive: {metrics['val_sensitive'][i]:.4f}\n")
            f.write(f" Validation Specificity: {metrics['val_specificity'][i]:.4f}\n")
            f.write(f" Learning Rate: {metrics['learning_rate'][i]:.6f}\n")
            f.write("\n")
        f.write(f"Total Model Parameters: {total_params}\n")
        f.write(f"Total Training Time: {total_train_time:.2f} seconds\n")
