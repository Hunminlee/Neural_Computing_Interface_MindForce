import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../Share')
import Model, utils, config



class TremorModelTrainer:
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
        else:
            print("subject must be Hunmin, Xianyu, Brian, Carlson")
            return

        self.classes = config.classes_5
        self.results = []
        self.model = None
        self.batch_size = config.batch_size
        self.epochs = config.epochs


    def Model_1D_CNN(self, input_shape):

        from tensorflow.keras import layers, models

        input_shape = (4, 1)  # (timesteps, features)

        model = models.Sequential([
            layers.Input(shape=input_shape),

            layers.Conv1D(8, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling1D(pool_size=2, padding='same'),

            layers.Conv1D(16, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling1D(pool_size=2, padding='same'),

            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(6, activation='softmax'),
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model



    def train_multiple_dataset_2D(self, X_train, y_train, X_test, y_test):
        self.model = Model.Original_model_V1(X_train.shape[1:])

        history, self.model = Model.Train_model(
            self.model, X_train, y_train, X_test, y_test,
            set_epoch=self.epochs, set_batch_size=self.batch_size, Model_name='V0',
            set_verbose=False, save_model_set=True
        )

        acc = self.model.evaluate(X_test, y_test, verbose=0)[1]
        print(f"Accuracy of test dataset using model V0: {acc * 100:.4f}%")
        return float(np.max(history.history['val_accuracy']) * 100), self.model


    def train_multiple_dataset_1D(self, X_train, y_train, X_test, y_test):
        self.model = self.Model_1D_CNN(X_train.shape[1:])

        history, self.model = Model.Train_model(
            self.model, X_train, y_train, X_test, y_test,
            set_epoch=self.epochs, set_batch_size=self.batch_size, Model_name='V0',
            set_verbose=False, save_model_set=False
        )

        acc = self.model.evaluate(X_test, y_test, verbose=0)[1]
        print(f"Accuracy of test dataset using model V0: {acc * 100:.4f}%")
        return float(np.max(history.history['val_accuracy']) * 100), self.model


    def save_results(self, filepath):
        df = pd.DataFrame({
            'Info': self.info_labels,
            'Info_Set': self.dataset_info,
            'Accuracy': self.results
        })
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}\n\n")
        return df


    def return_K_data(self, K, verbose=False):
        X_train_all, y_train_all = [], []
        X_test_all, y_test_all = [], []

        for idx, session_info in enumerate(self.dataset_info):
            if verbose:
                print(f"Dataset {idx + 1}/{len(self.dataset_info)} - Session {session_info}\n{'='*40}")
            path = os.path.join(self.default_path, f'{session_info}/raw/')

            if idx < K:
                feature_set, labels = utils.get_dataset(path, self.classes, show_labels=False)
                X_train, y_train, X_test, y_test = utils.split_data(feature_set, labels, ratio=0.99)

                # Stack cumulatively
                X_train_all.append(X_train)
                y_train_all.append(y_train)

                # Concatenate all so far
                X_train_stacked = np.concatenate(X_train_all, axis=0)
                y_train_stacked = np.concatenate(y_train_all, axis=0)

            else:
                feature_set, labels = utils.get_dataset(path, self.classes, show_labels=False)
                _, _, X_test, y_test = utils.split_data(feature_set, labels, ratio=0)

                # Stack cumulatively
                X_test_all.append(X_test)
                y_test_all.append(y_test)

                # Concatenate all so far
                X_test_stacked = np.concatenate(X_test_all, axis=0)
                y_test_stacked = np.concatenate(y_test_all, axis=0)

        print(X_train_stacked.shape, y_train_stacked.shape), X_test_stacked.shape, y_test_stacked.shape

        return X_train_stacked, y_train_stacked, X_test_stacked, y_test_stacked
