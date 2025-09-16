import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def heatmap_confusion_matrix_with_norm(cms, class_names, normalize):
    # Optionally normalize each confusion matrix
    cms_to_use = []
    for cm in cms:
        cm_np = cm.values if isinstance(cm, pd.DataFrame) else cm
        if normalize:
            row_sums = cm_np.sum(axis=1, keepdims=True)  # now works
            cm_norm = cm_np / row_sums
            cms_to_use.append(cm_norm)
        else:
            cms_to_use.append(cm_np)

    cms_to_use = np.array(cms_to_use)

    # Compute mean and std
    cm_mean = np.mean(cms_to_use, axis=0)
    cm_std = np.std(cms_to_use, axis=0)

    # Format annotations as mean ± std (2 decimals)
    annot = np.empty_like(cm_mean, dtype=object)
    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            annot[i, j] = f"{cm_mean[i, j]:.2f}\n±{cm_std[i, j]:.2f}"

    # Plot heatmap
    plt.figure(figsize=(6, 6), dpi=350)
    sns.heatmap(cm_mean, annot=annot, fmt='', cmap='Reds', cbar=False,
                xticklabels=class_names if class_names else range(cm_mean.shape[0]),
                yticklabels=class_names if class_names else range(cm_mean.shape[0]),
                annot_kws={"fontsize": 13})

    label_size = 15
    plt.xlabel('Predicted Label', fontsize=label_size)
    plt.ylabel('True Label', fontsize=label_size)
    title = 'Confusion Matrix (Normalized Mean ± Std)' if normalize else 'Confusion Matrix (Mean ± Std)'
    #plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.show()

    # Compute overall accuracy (mean of accuracies)
    if normalize:
        accs = [np.trace(cm) / cm.shape[0] for cm in cms_to_use]  # row-wise normalized, mean diag
    else:
        accs = [np.trace(cm) / np.sum(cm) for cm in cms_to_use]
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    print(f"Overall Accuracy: {mean_acc:.4f} ± {std_acc:.4f} "
          f"({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")

    return cm_mean, cm_std, mean_acc, std_acc



def heatmap_confusion_matrix(cms, class_names):
    cm_mean = np.mean(cms, axis=0)
    cm_std = np.std(cms, axis=0)

    # Format annotations as mean ± std (2 decimals)
    annot = np.empty_like(cm_mean, dtype=object)
    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            annot[i, j] = f"{cm_mean[i, j]:.1f}\n±({cm_std[i, j]:.1f})"

    # Plot heatmap
    plt.figure(figsize=(6, 6), dpi=350)
    sns.heatmap(cm_mean, annot=annot, fmt='', cmap='Reds', cbar=False,
                xticklabels=class_names if class_names else range(cm_mean.shape[0]),
                yticklabels=class_names if class_names else range(cm_mean.shape[0]),
                annot_kws={"fontsize": 13})

    label_size = 15
    plt.xlabel('Predicted Label', fontsize=label_size)
    plt.ylabel('True Label', fontsize=label_size)
    plt.title('Confusion Matrix (Mean ± Std)', fontsize=18)
    plt.tight_layout()
    plt.show()

    # Compute overall accuracy (mean of accuracies, not ratio of means)
    accs = [np.trace(cm) / np.sum(cm) for cm in cms]
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    print(f"Overall Accuracy: {mean_acc:.4f} ± {std_acc:.4f} "
          f"({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")

    return cm_mean, cm_std, mean_acc, std_acc