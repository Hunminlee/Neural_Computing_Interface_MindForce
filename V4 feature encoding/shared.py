import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



def feature_wise_test(X_train_stacked, y_train_stacked, X_test_stacked, y_test_stacked):

    feature_acc = []

    for idx, f in enumerate(feature_idx):
        X_train = X_train_stacked[:, :, f:f+1, :]
        X_test = X_test_stacked[:, :, f:f+1, :]

        X_train = np.squeeze(X_train, axis=-1)  # Remove last dim → (100000, 4, 1)
        X_test = np.squeeze(X_test, axis=-1)

        print(X_train.shape, X_test.shape)

        acc, _ = trainer.train_multiple_dataset_1D(X_train, y_train_stacked, X_test, y_test_stacked)
        feature_acc.append(acc)
        print("\n")

    return feature_acc


def feature_vis(feature_names, feature_acc):
    feature_names_idx = feature_names #+ ['All features']

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(feature_acc)), feature_acc, color='skyblue')
    plt.xticks(range(len(feature_acc)), feature_names_idx, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Feature-wise Accuracy')
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
    plt.show()