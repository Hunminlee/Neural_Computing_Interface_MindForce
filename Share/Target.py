import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import importlib
import warnings

warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model

import Model
#importlib.reload(Model)
import utils
import config


#Baseline : Training from scratch = Training in one data => This is our target performance

class ProgressiveTrainer:
    def __init__(self, config, subject):
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
        elif subject == "Harold":
            self.default_path = config.default_path_sub_H2
            self.info_labels = config.Info_sub_H2
            self.dataset_info = config.dataset_sub_H2
        else:
            print("subject must be Hunmin, Xianyu, Brian, Carlson, Harold")
            return

        self.classes = config.classes_5
        self.results = []
        self.model = None
        self.batch_size = config.batch_size
        self.epochs = config.epochs

        self.train_ratio = 0.5
        self.model_name = "Training_from_scratch"

        self.trained_acc_all = []
        self.prev_acc_all = []
        self.init_acc_all = []
        self.X_test_prev_all = []
        self.y_test_prev_all = []

    def get_result(self, path, first_time, plot_learning_curve):
        feature_set, labels = utils.get_dataset(path, self.classes, show_labels=False)
        X_train, y_train, X_test, y_test = utils.split_data(feature_set, labels, ratio=self.train_ratio)

        model = Model.Original_model_V1(X_train.shape[1:])
        init_acc = model.evaluate(X_test, y_test, verbose=0)[1]

        history, _ = Model.Train_model(
            model, X_train, y_train, X_test, y_test,
            self.epochs, self.batch_size, self.model_name,
            set_verbose=0, save_model_set=False
        )

        if plot_learning_curve:
            utils.visualize_history(history)

        test_acc = float(np.max(history.history['val_accuracy']))

        if first_time:
            self.X_test_prev_all = X_test
            self.y_test_prev_all = y_test
        else:
            self.X_test_prev_all = np.concatenate((self.X_test_prev_all, X_test), axis=0)
            self.y_test_prev_all = np.concatenate((self.y_test_prev_all, y_test), axis=0)

        prev_acc = model.evaluate(self.X_test_prev_all, self.y_test_prev_all, verbose=0)[1]

        return init_acc, test_acc, prev_acc

    def run(self, plot_learning_curve=False):
        for idx, session in enumerate(self.dataset_info):
            print(f"\n{'='*43}\nDataset {idx+1}/{len(self.dataset_info)} - Session {session}\n{'='*43}")
            path = os.path.join(self.default_path, session + 'raw/')
            first_time = (idx == 0)
            init_acc, test_acc, prev_acc = self.get_result(path, first_time, plot_learning_curve)

            self.init_acc_all.append(init_acc)
            self.trained_acc_all.append(test_acc)
            self.prev_acc_all.append(prev_acc)

        return self.init_acc_all, self.trained_acc_all, self.prev_acc_all


    def plot_results(self, baselines, baseline_K):

        plt.figure(figsize=(15, 8))
        plt.title('Training from scratch every time', fontsize=15)
        #plt.plot(self.info_labels, self.init_acc_all, marker='o', label='Untrained model on current data', linestyle='--')
        plt.plot(self.info_labels, [1/6 for i in range(len(self.info_labels))], marker='o', label='Untrained model on current data', linestyle='--')
        plt.plot(self.info_labels, self.prev_acc_all, marker='o', label='Model trained + tested on accumulated data')
        plt.plot(self.info_labels, self.trained_acc_all, marker='o', label='Model trained + tested on current split')

        for idx, base in enumerate(baselines):
            baseline_result = pd.read_csv(base)
            plt.plot(self.info_labels, baseline_result['Accuracy'] / 100, marker='^', label=f'Baseline V{idx} - K:{baseline_K[idx]}', linestyle='--')

        plt.ylim([0, 1])
        plt.xlabel('Date (Sessions)')
        plt.ylabel('Test Accuracy')
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()