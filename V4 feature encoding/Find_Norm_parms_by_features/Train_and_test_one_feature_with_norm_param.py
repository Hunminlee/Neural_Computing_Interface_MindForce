import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import Processing_same_with_MATLAB
import os
import matplotlib.pyplot as plt
import config, scipy, utils
import Trainer


class main:

    def __init__(self, filter_a, filter_b, data_files, default_path, SUBJECT, K):

        self.filter_a = filter_a
        self.filter_b = filter_b
        self.data_files = data_files
        self.default_path = default_path
        self.trainer = Trainer.TremorModelTrainer(config, subject=SUBJECT)
        self.K = K

    #1D - one feature at a time
    def Train_and_test(self, Normalization_TF, mean, std, num_feature_set, target_feat_idx):

        extractor = Processing_same_with_MATLAB.EMGFeatureExtractor(mean, std, self.filter_b, self.filter_a, Norm_bool=Normalization_TF, num_feature_set=num_feature_set)
        X_train_all, y_train_all, X_test_all, y_test_all, X_val_all, y_val_all = [], [], [], [], [], []
        unseen_test_result = []

        for idx, session_info in enumerate(self.data_files):
            print(f"Dataset {idx + 1}/{len(self.data_files)} - Session {session_info}\n{'='*40}")
            path = os.path.join(self.default_path, f'{session_info}raw/')
            features, class_labels = [], []
            for c_idx, c in enumerate(config.classes_5):
                raw_data = os.listdir(path+c)
                mat = scipy.io.loadmat(path+c+raw_data[0])
                extractor.buffer = mat['Data_ADC']
                class_labels.append(mat['Data_Cls'].reshape(-1))

                #### features_per_cls = extractor.extract_features(num_feature_set=num_feature_set)  ### 이건 2D input
                features_per_cls = extractor.extract_one_feature_at_a_time(target_feature_idx=target_feat_idx)  ###1D input

                #여기서 Normalization_TF
                #features = (features - self.feat_mean[:, :, np.newaxis]) / self.feat_std[:, :, np.newaxis]
                features_per_cls = extractor.Normalization(features_per_cls, mean, std, target_feat_idx)

                features_per_cls = np.transpose(features_per_cls, (1, 0))  # shape: (1729, 4, 14)
                features.append(features_per_cls)
                #print(features_per_cls.shape, mat['Data_Cls'].reshape(-1).shape)

            X = np.concatenate(features, axis=0)
            y = np.concatenate(class_labels, axis=0)
            if X.shape[0] != y.shape[-1]:
                print(f"Incorrect shape between features and Class: {X.shape} and {y.shape}, {session_info}")
                break

            if idx < self.K:
                X_train, y_train, X_val, y_val = utils.split_data(X, y, ratio=0.8)
                X_train_all.append(X_train)
                y_train_all.append(y_train)
                X_val_all.append(X_train)
                y_val_all.append(y_train)

            elif idx == self.K:
                X_train, y_train, X_test, y_test,  = utils.split_data(X, y, ratio=0.8)
                X_train_all.append(X_train)
                y_train_all.append(y_train)
                X_val_all.append(X_test)
                y_val_all.append(y_test)

                X_train_stacked = np.concatenate(X_train_all, axis=0)
                y_train_stacked = np.concatenate(y_train_all, axis=0)
                print(f"\t Training {self.K}: ", X_train_stacked.shape, y_train_stacked.shape)
                acc, pre_trained_CNN = self.trainer.train_multiple_dataset_1D(X_train, y_train, X_test, y_test)
                print(f"\t Accuracy on test dataset {idx+1}: {acc:.4f}%")

            else:
                X_test, y_test, _, _ = utils.split_data(X, y, ratio=1)
                X_test_all.append(X_test)
                y_test_all.append(y_test)
                X_test_stacked = np.concatenate(X_test_all, axis=0)
                y_test_stacked = np.concatenate(y_test_all, axis=0)

                X = np.expand_dims(X, axis=-1)
                acc = pre_trained_CNN.evaluate(X_test, y_test, verbose=0)[1]*100
                print(f"\t Accuracy on unseen dataset {idx+1}: {acc:.4f}%")
                unseen_test_result.append(acc)

        return unseen_test_result, X_train_stacked, y_train_stacked, X_test_stacked, y_test_stacked