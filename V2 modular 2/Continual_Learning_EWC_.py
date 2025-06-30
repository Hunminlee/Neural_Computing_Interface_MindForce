import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

import utils
import Model
import config


class ContinualLearningTrainer:
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

        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.train_ratio = 0.5
        self.model_name = "Cont_L_Model"

        self.init_acc_all = []
        self.prev_acc_all = []
        self.trained_acc_all = []
        self.X_test_prev_all = []
        self.y_test_prev_all = []

    def init_stage(self, X_train, y_train, model):
        weights_task = [tf.identity(var) for var in model.trainable_variables]
        importance = [tf.zeros_like(var) for var in model.trainable_variables]
        batch = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.batch_size)

        for x_batch, y_batch in batch:
            with tf.GradientTape() as tape:
                preds = model(x_batch)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            for i, grad in enumerate(grads):
                if grad is not None:
                    importance[i] += tf.square(grad)

        importance = [imp / len(batch) for imp in importance]
        return weights_task, importance

    def adaptation_stage(self, X_train, y_train, X_test, y_test, model, weights_prev_task, importance):
        lambda_ewc = 1000.0
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.batch_size)

        acc_init = model.evaluate(X_test, y_test, verbose=0)[1]

        for epoch in range(self.epochs):
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    preds = model(x_batch, training=True)
                    loss = loss_fn(y_batch, preds)
                    for var, old_w, imp in zip(model.trainable_variables, weights_prev_task, importance):
                        loss += (lambda_ewc / 2) * tf.reduce_sum(imp * tf.square(var - old_w))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        acc_prev_data = model.evaluate(self.X_test_prev_all, self.y_test_prev_all, verbose=0)[1]
        acc_current_stage = model.evaluate(X_test, y_test, verbose=0)[1]

        return acc_init, acc_prev_data, acc_current_stage

    def run(self):
        for idx, session in enumerate(self.dataset_info):
            #print(f"{'='*43}\nDataset {idx+1}/{len(self.dataset)} - Session {session}\n{'='*43}")

            print(f"Dataset {idx + 1}/{len(self.dataset_info)} - Session {session}\n{'=' * 43}")
            path = os.path.join(self.default_path, session, 'raw/')

            feature_set, labels = utils.get_dataset(path, self.classes, show_labels=False)
            X_train, y_train, X_test, y_test = utils.split_data(feature_set, labels, ratio=self.train_ratio)

            if idx == 0:
                model = Model.Original_model_V1(X_train.shape[1:])
                history, model = Model.Train_model(
                    model, X_train, y_train, X_test, y_test,
                    self.epochs, self.batch_size, self.model_name,
                    set_verbose=0, save_model_set=True
                )
                acc = model.evaluate(X_test, y_test, verbose=0)[1]
                self.init_acc_all, self.prev_acc_all, self.trained_acc_all = [0.2], [acc], [acc]
                self.X_test_prev_all, self.y_test_prev_all = X_test, y_test
            else:
                model = load_model(f'{self.model_name}.keras')
                weights_task, importance = self.init_stage(X_train, y_train, model)
                init, prev, current = self.adaptation_stage(
                    X_train, y_train, X_test, y_test, model,
                    weights_task, importance
                )
                self.init_acc_all.append(init)
                self.prev_acc_all.append(prev)
                self.trained_acc_all.append(current)
                self.X_test_prev_all = np.concatenate((self.X_test_prev_all, X_test), axis=0)
                self.y_test_prev_all = np.concatenate((self.y_test_prev_all, y_test), axis=0)
                del model

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
