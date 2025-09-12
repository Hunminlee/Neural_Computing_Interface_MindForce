import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)


class MetaLearner:
    def __init__(self, input_model, N_way, input_shape, meta_iters, meta_step_size):

        self.meta_iters = meta_iters
        self.meta_step_size = meta_step_size
        self.num_classes = N_way
        self.input_shape = input_shape
        self.model = input_model


    def train(self, X_train, y_train, X_test, y_test, get_data_Meta, N_way, K_shot):
        accuracies = []

        for meta_iter in range(self.meta_iters):
            frac_done = meta_iter / self.meta_iters
            cur_step_size = (1 - frac_done) * self.meta_step_size
            old_weights = self.model.get_weights()
            train_data, test_data, proto_X, proto_y = get_data_Meta(X_train, y_train, X_test, y_test, N_way=N_way, K_shot=K_shot, split=True)

            if train_data is None or test_data is None or len(train_data[0]) == 0 or len(test_data[0]) == 0:
                print(f"⚠️ Skipping iteration {meta_iter+1} due to empty train/test data.")
                continue

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            # Step 3: Train on one episode
            result = self.model.fit(train_data[0], train_data[1], epochs=5, validation_data=(test_data[0], test_data[1]), verbose=0)
            val_acc = np.max(result.history['val_accuracy'])
            accuracies.append(val_acc)

            # Step 4: Meta-update
            new_weights = self.model.get_weights()
            updated_weights = [
                old + (new - old) * cur_step_size
                for old, new in zip(old_weights, new_weights)
            ]
            self.model.set_weights(updated_weights)

        return np.max(accuracies)


    def train_for_heatmap(self, X_train, y_train, X_test, y_test, get_data_Meta, N_way, K_shot):
        accuracies = []

        for meta_iter in range(self.meta_iters):
            frac_done = meta_iter / self.meta_iters
            cur_step_size = (1 - frac_done) * self.meta_step_size
            old_weights = self.model.get_weights()
            train_data, test_data, proto_X, proto_y = get_data_Meta(X_train, y_train, X_test, y_test, N_way=N_way,
                                                                    K_shot=K_shot, split=True)

            if train_data is None or test_data is None or len(train_data[0]) == 0 or len(test_data[0]) == 0:
                print(f"⚠️ Skipping iteration {meta_iter + 1} due to empty train/test data.")
                continue

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            # Step 3: Train on one episode
            result = self.model.fit(train_data[0], train_data[1], epochs=5, validation_data=(test_data[0], test_data[1]),
                                    verbose=0)
            val_acc = np.max(result.history['val_accuracy'])
            accuracies.append(val_acc)

            # Step 4: Meta-update
            new_weights = self.model.get_weights()
            updated_weights = [
                old + (new - old) * cur_step_size
                for old, new in zip(old_weights, new_weights)
            ]
            self.model.set_weights(updated_weights)


        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        return np.max(accuracies), y_pred




def get_data_Meta(X_train, y_train, X_test, y_test, N_way, K_shot, split=True):

    unique_classes = np.unique(y_train)
    selected_classes = np.random.choice(unique_classes, size=N_way, replace=False)

    X_support, y_support = [], []
    X_query, y_query = [], []

    for cls in selected_classes:
        idxs = np.where(y_train == cls)[0]
        np.random.shuffle(idxs)
        if len(idxs) < K_shot * 2:
            print("HERE")
            continue

        support_idxs = idxs[:K_shot]
        query_idxs = idxs[K_shot:K_shot*2]  # equal size query set
        #query_idxs = idxs[K_shot:]  # equal size query set

        X_support.append(X_train[support_idxs])
        y_support.append(y_train[support_idxs])

        if split:
            X_query.append(X_train[query_idxs])
            y_query.append(y_train[query_idxs])

    X_support = np.concatenate(X_support, axis=0)
    y_support = np.concatenate(y_support, axis=0)
    perm = np.random.permutation(len(X_support))
    X_support, y_support = X_support[perm], y_support[perm]

    if split:
        X_query = np.concatenate(X_query, axis=0)
        y_query = np.concatenate(y_query, axis=0)
        perm = np.random.permutation(len(X_query))
        X_query, y_query = X_query[perm], y_query[perm]

        return (X_support, y_support), (X_query, y_query), X_support, y_support
    else:
        return X_support, y_support
