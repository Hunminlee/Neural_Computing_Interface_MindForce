import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

import Model
import utils
import config
importlib.reload(Model)


class TransferLearningTrainer:
    def __init__(self, config, subject, increment_true_false):
        if subject == "Hunmin":
            self.default_path = config.default_path_sub_H
            self.info_labels = config.Info_sub_H
            self.dataset_info = config.dataset_sub_H
        elif subject == "Xianyu":
            self.default_path = config.default_path_sub_X
            self.info_labels = config.Info_sub_X
            self.dataset_info = config.dataset_sub_X
        elif subject == "Brian":
            self.default_path = config.default_path_sub_B
            self.info_labels = config.Info_sub_B
            self.dataset_info = config.dataset_sub_B
        elif subject == "Carlson":
            self.default_path = config.default_path_sub_C
            self.info_labels = config.Info_sub_C
            self.dataset_info = config.dataset_sub_C
        else:
            print("subject must be Hunmin, Xianyu, Brian, Carlson")
            return

        self.classes = config.classes_5
        self.train_ratio = 0.5
        self.set_epoch = config.epochs
        self.set_batch_size = config.batch_size
        self.model_name = "TL_Model_increment"
        self.TL_increment = increment_true_false

        self.init_acc_all = []
        self.prev_acc_all = []
        self.trained_acc_all = []
        self.X_test_prev_all = []
        self.y_test_prev_all = []

    def train_initial_model(self, path):
        feature_set, labels = utils.get_dataset(path, self.classes, show_labels=False)
        X_train, y_train, X_test, y_test = utils.split_data(feature_set, labels, ratio=self.train_ratio)

        model = Model.Original_model_V1(X_train.shape[1:])
        history, model = Model.Train_model(
            model, X_train, y_train, X_test, y_test,
            self.set_epoch, self.set_batch_size, self.model_name,
            set_verbose=0, save_model_set=True
        )

        acc = model.evaluate(X_test, y_test, verbose=0)[1]
        self.init_acc_all.append(0.2)  # Dummy initial value for consistency
        self.prev_acc_all.append(acc)
        self.trained_acc_all.append(acc)
        self.X_test_prev_all = X_test
        self.y_test_prev_all = y_test

    def adapt_model(self, path, plot_learning_curve):
        feature_set, labels = utils.get_dataset(path, self.classes, show_labels=False)
        X_train, y_train, X_test, y_test = utils.split_data(feature_set, labels, self.train_ratio)

        model = load_model(f'{self.model_name}.keras')
        init_test_loss, init_test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nInitial Test Accuracy: {init_test_acc*100:.2f}%")

        history, adapted_model = Model.Train_model(
            model, X_train, y_train, X_test, y_test,
            self.set_epoch, self.set_batch_size, self.model_name,
            set_verbose=0, save_model_set=self.TL_increment
        )

        if plot_learning_curve:
            utils.visualize_history(history)

        self.X_test_prev_all = np.concatenate((self.X_test_prev_all, X_test), axis=0)
        self.y_test_prev_all = np.concatenate((self.y_test_prev_all, y_test), axis=0)

        prev_acc = adapted_model.evaluate(self.X_test_prev_all, self.y_test_prev_all, verbose=0)[1]
        max_test_acc = np.max(history.history['val_accuracy'])
        acc_diff = (max_test_acc - init_test_acc) * 100

        print(f"Accuracy Improvement: {acc_diff:.2f}%")
        print("\t ===> Positive" if acc_diff > 0 else "\t ===> Negative")

        self.init_acc_all.append(init_test_acc)
        self.prev_acc_all.append(prev_acc)
        self.trained_acc_all.append(max_test_acc)

    def run(self, plot_learning_curve=False):
        for idx, session in enumerate(self.dataset_info):
            print(f"\n{'='*43}\nDataset {idx+1}/{len(self.dataset_info)} - Session {session}\n{'='*43}")
            path = os.path.join(self.default_path, session, 'raw/')
            if idx == 0:
                self.train_initial_model(path)
            else:
                self.adapt_model(path, plot_learning_curve)

        return self.init_acc_all, self.trained_acc_all, self.prev_acc_all

    def plot_results(self, baselines, baseline_K):

        plt.figure(figsize=(15, 8))
        plt.title('Training from scratch every time', fontsize=15)
        plt.plot(self.info_labels, self.init_acc_all, marker='o', label='Untrained model on current data', linestyle='--')
        plt.plot(self.info_labels, self.prev_acc_all, marker='o', label='Model trained + tested on accumulated data')
        plt.plot(self.info_labels, self.trained_acc_all, marker='o', label='Model trained + tested on current split')

        for idx, base in enumerate(baselines):
            baseline_result = pd.read_csv(base)
            plt.plot(self.info_labels, baseline_result['Accuracy'] / 100, marker='^', label=f'Baseline V{idx} - K:{baseline_K[idx]}', linestyle='--')

        plt.ylim([0, 1])
        plt.xlabel('Date (Sessions)')
        plt.ylabel('Test Accuracy')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
