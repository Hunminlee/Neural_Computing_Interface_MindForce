import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import utils
import config

class MAML(tf.keras.Model):
    def __init__(self, input_shape, num_classes, inner_lr=0.01, outer_lr=0.001):
        super(MAML, self).__init__()
        self.inner_lr = inner_lr
        self.outer_optimizer = tf.keras.optimizers.Adam(learning_rate=outer_lr)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Define a simple MLP model as an example (can be replaced with CNN)
        self.base_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])

    def clone_model_with_weights(self):
        cloned_model = tf.keras.models.clone_model(self.base_model)
        cloned_model.set_weights(self.base_model.get_weights())
        return cloned_model

    def adapt(self, model, x_train, y_train):
        with tf.GradientTape() as tape:
            logits = model(x_train, training=True)
            loss = self.loss_fn(y_train, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        k = 0
        for var in model.trainable_variables:
            var.assign_sub(self.inner_lr * grads[k])
            k += 1
        return model

    def train_step(self, support_set, query_set):
        x_support, y_support = support_set
        x_query, y_query = query_set

        cloned_model = self.clone_model_with_weights()
        adapted_model = self.adapt(cloned_model, x_support, y_support)

        with tf.GradientTape() as tape:
            logits = adapted_model(x_query, training=True)
            loss = self.loss_fn(y_query, logits)
        #grads = tape.gradient(loss, self.base_model.trainable_variables)
        grads = [(a - b) / self.inner_lr for a, b in zip(self.base_model.trainable_variables, adapted_model.trainable_variables)]

        self.outer_optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))

        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y_query), tf.float32))
        return loss, acc

    def evaluate(self, x, y):
        logits = self.base_model(x)
        pred = tf.argmax(logits, axis=1)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))
        return acc.numpy()



class MAMLProgressiveTrainer:
    def __init__(self, config, subject, K_shot, query_size):
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
        self.inner_epochs = 1
        self.outer_epochs = config.epochs
        self.query_size = query_size
        self.shot = K_shot

        self.trained_acc_all = []
        self.prev_acc_all = []
        self.init_acc_all = []

        self.X_test_prev_all = []
        self.y_test_prev_all = []


    def prepare_episode(self, X, y, n_shot, n_query):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]
        return (X[:n_shot].astype(np.float32), y[:n_shot]), (X[n_shot:n_shot+n_query].astype(np.float32), y[n_shot:n_shot+n_query])


    def run(self):
        for idx, session in enumerate(self.dataset_info):
            print(f"Dataset {idx+1}/{len(self.dataset_info)} - Session {session}\n{'='*43}")
            path = os.path.join(self.default_path, session + 'raw/')
            feature_set, labels = utils.get_dataset(path, self.classes, show_labels=False)
            X_train, y_train, X_test, y_test = utils.split_data(feature_set, labels, ratio=self.train_ratio)

            if idx == 0:
                input_shape = X_train.shape[1:]
                num_classes = len(np.unique(y_train))
                self.maml = MAML(input_shape=input_shape, num_classes=num_classes)

                support, query = self.prepare_episode(X_train, y_train, self.shot, self.query_size)

                for epoch in range(self.outer_epochs):
                    loss, acc = self.maml.train_step(support, query)
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")

                acc = self.maml.evaluate(X_test, y_test)
                self.init_acc_all.append(0.2)
                self.trained_acc_all.append(acc)
                self.prev_acc_all.append(acc)
                self.X_test_prev_all = X_test
                self.y_test_prev_all = y_test
            else:
                support, query = self.prepare_episode(X_train, y_train, self.shot, self.query_size)

                for epoch in range(self.outer_epochs):
                    loss, acc = self.maml.train_step(support, query)

                acc_curr = self.maml.evaluate(X_test, y_test)
                X_combined = np.concatenate((self.X_test_prev_all, X_test), axis=0)
                y_combined = np.concatenate((self.y_test_prev_all, y_test), axis=0)
                acc_prev = self.maml.evaluate(X_combined, y_combined)

                self.init_acc_all.append(0.2)
                self.trained_acc_all.append(acc_curr)
                self.prev_acc_all.append(acc_prev)
                self.X_test_prev_all = X_combined
                self.y_test_prev_all = y_combined

        return self.init_acc_all, self.prev_acc_all, self.trained_acc_all

    def plot_results(self, baselines, baseline_K):

        plt.figure(figsize=(15, 8))
        plt.title('Meta-training using few-shot samples', fontsize=15)
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