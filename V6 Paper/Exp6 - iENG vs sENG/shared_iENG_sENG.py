import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import scipy, os
from sklearn.metrics import confusion_matrix
import seaborn as sns

import sys

from sklearn.model_selection import learning_curve

sys.path.append('../../Share/')
sys.path.append('../../Share/Manual_processing/')
import baseline, config, Model, utils, Same_with_MATLAB, Feature_info



def restore_labels(mat, labels_windowed):

    original_length = mat['Data_ADC'].shape[1]
    win_size = 2000 # 200ms 10
    win_step = 100
    #valid_length = original_length - 2 * 60

    label_full = np.zeros(original_length, dtype=labels_windowed.dtype) # 복원될 시계열 레이블 (원본 길이)

    # 슬라이딩 윈도우 인덱스 따라 레이블 채워넣기
    for i, label in enumerate(labels_windowed):
        start = 60 + i * win_step
        end = start + win_size
        if end <= original_length - 60:
            label_full[start:end] = label

    return label_full


def filtering_zero(X, y, erase_label):
    # 1. erase_label 제거
    keep_indices = y != erase_label
    X = X[keep_indices]
    y = y[keep_indices]

    # 2. erase_label보다 큰 값은 1씩 감소
    y = np.where(y > erase_label, y - 1, y)

    return X, y

def vis_graph(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()

def heatmap_confusion_matrix(X_test, y_test, model):
    # Predict class labels on the test set
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)  # convert softmax probs to predicted class
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(np.max(y_test)+1), yticklabels=range(np.max(y_test)+1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3,4,5], yticklabels=[0,1,2,3,4,5])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    return cm


from collections import Counter

def balance_data(X, y):
    # Count samples per class
    class_counts = Counter(y)
    min_count = min(class_counts.values())  # target: balance all to minority count

    indices_list = []

    for label in sorted(class_counts.keys()):
        label_indices = np.where(y == label)[0]
        selected_indices = np.random.choice(label_indices, size=min_count, replace=False)
        indices_list.extend(selected_indices)

    # Shuffle all selected indices
    balanced_indices = np.random.permutation(indices_list)

    # Subset the data
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    return X_balanced, y_balanced



def return_X_y(path, num_session, balance):
    fs, lower_cutoff, upper_cutoff = Feature_info.fs, Feature_info.lower_cutoff, Feature_info.upper_cutoff
    # fs, lower_cutoff, upper_cutoff = Feature_info.fs, 1, 300
    filter_b, filter_a = Same_with_MATLAB.cheby2(4, 30, [lower_cutoff / (fs/2), upper_cutoff / (fs/2)], btype='bandpass')

    data_per_class_files = os.listdir(path)
    X, y = [], []

    for f in num_session:
        for cls in data_per_class_files:
            input_path = path+cls+'/'
            files = os.listdir(input_path)
            mat = scipy.io.loadmat(input_path+files[f])
            label = mat['Data_Cls'].reshape(-1)  # shape: (1, 1729)

            feat_mean = np.tile(Feature_info.feat_mean_lst, (4, 1))
            feat_std = np.tile(Feature_info.feat_std_lst, (4, 1))

            mapped_label = np.where(label == 0, 0, int(cls))
            restored_label = restore_labels(mat, mapped_label)

            #print(mat['Data_ADC'].shape, mat['Data_Cls'].shape, restored_label.shape)
            extractor = Same_with_MATLAB.EMGFeatureExtractor(feat_mean, feat_std, filter_b, filter_a, Norm_bool=False, num_feature_set=14) #I tried 23, but not so good
            extractor.buffer = mat['Data_ADC']
            #1000, 50 = winsize and winstep
            features, labels = extractor.extract_features_with_labels(win_size=2000, win_step=100, feat_exclude=25, filtering=False, restored_label=restored_label)

            features = np.transpose(features, (2, 0, 1))  # shape: (1729, 4, 14)
            X.append(features)
            y.append(labels)
            #print(features.shape, labels.shape)

    X_train = np.concatenate(X, axis=0)
    y_train = np.concatenate(y, axis=0)
    X_train = X_train[:, :, :, np.newaxis]
    #print(pd.Series(y_train).value_counts())
    print(X_train.shape, y_train.shape)

    if balance:
        X_train, y_train = balance_data(X_train, y_train)

    return X_train, y_train


def return_X_y_get_from_matfile(path, num_session, balance):
    data_per_class_files = os.listdir(path)
    X, y = [], []

    for f in num_session:
        for cls in data_per_class_files:
            input_path = path+cls+'/'
            files = os.listdir(input_path)
            mat = scipy.io.loadmat(input_path+files[f])
            labels = mat['Data_Cls'].reshape(-1)  # shape: (1, 1729)

            features = mat['Data_Fea']
            features = np.transpose(features, (2, 0, 1))  # shape: (1729, 4, 14)
            X.append(features)
            y.append(labels)

    X_train = np.concatenate(X, axis=0)
    y_train = np.concatenate(y, axis=0)
    X_train = X_train[:, :, :, np.newaxis]
    #print(pd.Series(y_train).value_counts())
    print(X_train.shape, y_train.shape)

    if balance:
        X_train, y_train = balance_data(X_train, y_train)

    return X_train, y_train


def X_y_from_matfile(path, balance, modality, session):

    X, y = [], []

    if modality=='EMG':
        bluetooth_id = 'E8DD80E550BB'
    elif modality=='ENG':
        bluetooth_id = 'E9AD0E7DCC2B'
    else:
        print("modality must be EMG or ENG")
        return 0

    if session=='v1' or session=='v2':
        pass
    else:
        print("session must be v1 or v2")
        return 0

    data_per_class_files = os.listdir(path+f'{modality}_{session}/{bluetooth_id}/raw/')
    #print(data_per_class_files)

    for cls in data_per_class_files:
        input_path = path+f'{modality}_{session}/{bluetooth_id}/raw/{cls}/'
        files = os.listdir(input_path)

        mat = scipy.io.loadmat(input_path+files[0])
        labels = mat['Data_Cls'].reshape(-1)  # shape: (1, 1729)
        if int(cls) > 5:
            labels = [int(cls) if x == 5 else x for x in labels]

        features = mat['Data_Fea']
        features = np.transpose(features, (2, 0, 1))  # shape: (1729, 4, 14)
        X.append(features)
        y.append(labels)

    X_train = np.concatenate(X, axis=0)
    y_train = np.concatenate(y, axis=0)
    X_train = X_train[:, :, :, np.newaxis]
    #print(pd.Series(y_train).value_counts())
    #print(X_train.shape, y_train.shape)

    if balance:
        X_train, y_train = balance_data(X_train, y_train)

    return X_train, y_train



def train_model_feature_wise(X_train, y_train, X_test, y_test):
    X_train, y_train = filtering_zero(X_train, y_train, erase_label=0)
    X_test, y_test = filtering_zero(X_test, y_test, erase_label=0)
    ACC_lst = []

    for feature_idx in [2, 10]:
        One_X_train = X_train[:, :, feature_idx, :]
        One_X_test = X_test[:, :, feature_idx, :]

        model = Model.Original_model_1DCNN(One_X_train.shape[1:], num_class=np.max(y_train)+1)

        history, model = Model.Train_model(
            model, One_X_train, y_train, One_X_test, y_test,
            set_epoch=200, set_batch_size=256, Model_name='V0',
            set_verbose=False, save_model_set=False
        )
        ACC_lst.append(np.max(history.history['val_accuracy']))
        #vis_graph(history)
        #print("\n\n")
        #heatmap_confusion_matrix(One_X_test, y_test, model)

def train_model(X_train, y_train, X_test, y_test, heatmap_bool=False, draw_learning_curve=False):
    #X_train, y_train = filtering_zero(X_train, y_train, erase_label=0)
    #X_test, y_test = filtering_zero(X_test, y_test, erase_label=0)

    model = Model.Original_model(X_train.shape[1:], num_class=np.max(y_train)+1)

    history, model = Model.Train_model(
        model, X_train, y_train, X_test, y_test,
        set_epoch=100, set_batch_size=256, Model_name='V0',
        set_verbose=False, save_model_set=False
    )
    if draw_learning_curve:
        vis_graph(history)

    if heatmap_bool:
        cm = heatmap_confusion_matrix(X_test, y_test, model)
        return cm
    return history

def run(subject, Train_session, Test_session):
    bluetooth_id = 'E9AD0E7DCC2B'

    X_train, y_train = return_X_y_get_from_matfile(path = base_path+f'{subject}/{bluetooth_id}/raw/', num_session=Train_session, balance=True)
    X_test, y_test = return_X_y_get_from_matfile(path = base_path+f'{subject}/{bluetooth_id}/raw/', num_session=Test_session, balance=True)

    cm = train_model(X_train, y_train, X_test, y_test, heatmap_bool=True, draw_learning_curve=False)
    return cm
    #train_model_feature_wise(X_train, y_train, X_test, y_test)


def get_X_y(X_lst, y_lst, session, balance=True):
    X, y = [], []
    for t in session:
        X_tmp, y_tmp = X_lst[t].reshape(-1,16,14,1), y_lst[t]
        X.append(X_tmp)
        y.append(y_tmp)

    X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    if balance:
        X,y = balance_data(X, y)
    return X, y