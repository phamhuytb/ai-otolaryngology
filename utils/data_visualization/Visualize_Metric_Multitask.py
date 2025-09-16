import json
import matplotlib.pyplot as plt

def plot_metric(metric, save2image=False, path=None):

    loss_train = metric['train']['loss']
    loss_train_ent = metric['train']['loss ent']
    loss_train_diseases = metric['train']['loss diseases']
    accuracy_train = metric['train']['accuracy']
    f1_score_train = metric['train']['f1 score']

    loss_val = metric['val']['loss']
    loss_val_ent = metric['val']['loss ent']
    loss_val_diseases = metric['val']['loss diseases']
    accuracy_val = metric['val']['accuracy']
    f1_score_val = metric['val']['f1 score']

    # num epoch
    epochs = list(range(1, len(loss_train) + 1))

    # subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    # LOSS
    axs[0].plot(epochs, loss_train, marker='o', markersize=3, linestyle='-', color='b', label='train')
    axs[0].plot(epochs, loss_val, marker='o', markersize=3, linestyle='-', color='g', label='val')
    axs[0].plot(epochs, loss_train_ent, marker='o', markersize=3, linestyle='-', color='r', label='train ent')
    axs[0].plot(epochs, loss_train_diseases, marker='o', markersize=3, linestyle='-', color='c', label='train diseases')
    axs[0].plot(epochs, loss_val_ent, marker='o', markersize=3, linestyle='-', color='m', label='val ent')
    axs[0].plot(epochs, loss_val_diseases, marker='o', markersize=3, linestyle='-', color='y', label='val diseases')
    axs[0].set_title('LOSS')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid(False)
    axs[0].legend()
    axs[0].set_xticks(epochs)

    # ACCURACY
    axs[1].plot(epochs, accuracy_train, marker='o', markersize=3, linestyle='-', color='b', label='train')
    axs[1].plot(epochs, accuracy_val, marker='o', markersize=3, linestyle='-', color='g', label='val')
    axs[1].set_title('ACCURACY')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(False)
    axs[1].legend()
    axs[1].set_xticks(epochs)

    # F1 SCORE
    axs[2].plot(epochs, f1_score_train, marker='o', markersize=3, linestyle='-', color='b', label='train')
    axs[2].plot(epochs, f1_score_val, marker='o', markersize=3, linestyle='-', color='g', label='val')
    axs[2].set_title('F1-SCORE')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('F1-Score')
    axs[2].grid(False)
    axs[2].legend()
    axs[2].set_xticks(epochs)


    plt.tight_layout()

    # Save to image
    if save2image:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


# file_path = '/home/bht/Downloads/Output/metric.json'
# # Mở và đọc tệp JSON
# with open(file_path, 'r', encoding='utf-8') as file:
#     metric = json.load(file)
# plot_metric(metric, mode='train')
