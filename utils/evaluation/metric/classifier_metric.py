from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score,confusion_matrix

def compute_accuracy(true_labels, pred_labels):

    return accuracy_score(true_labels, pred_labels)

def compute_f1_score(true_labels, pred_labels, average='macro'):

    return f1_score(true_labels, pred_labels, average=average)

def compute_precision(true_labels, pred_labels, average='macro'):

    return precision_score(true_labels, pred_labels, average=average)

def compute_sensitive(true_labels, pred_labels, average='macro'):
    return recall_score(true_labels, pred_labels, average=average)
import numpy as np

def compute_specificity(true_labels, pred_labels, average='macro'):
    cm = confusion_matrix(true_labels, pred_labels)
    tn = np.diag(cm)  # True negatives are on the diagonal
    fp = cm.sum(axis=0) - tn  # False positives are the column sums minus the diagonal
    fn = cm.sum(axis=1) - tn  # False negatives are the row sums minus the diagonal
    tp = cm.sum() - (fp + fn + tn)  # True positives are total minus fp, fn, and tn

    specificity = tn / (tn + fp)
    if average == 'macro':
        return np.mean(specificity)
    elif average == 'weighted':
        support = cm.sum(axis=1)
        return np.average(specificity, weights=support)
    else:
        return specificity
