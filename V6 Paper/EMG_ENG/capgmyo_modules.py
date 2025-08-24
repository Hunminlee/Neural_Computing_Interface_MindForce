import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import random
import warnings

from scipy.signal import welch
import seaborn as sns

warnings.filterwarnings("ignore")

import gc
gc.collect()



def wavelength_feature(X, win_len, win_inc):

    T, C = X.shape
    n_windows = (T - win_len) // win_inc + 1
    WL_features = np.zeros((n_windows, C))

    for w in range(n_windows):
        start = w * win_inc
        end = start + win_len
        window = X[start:end, :]  # shape (win_len, C)

        # WL for each channel
        WL = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
        WL_features[w, :] = WL

    return WL_features

def import_dataset_train(subject, rep, path, win_len, win_inc, sel_four_ch, TEST_REP = 10):
    file_list = os.listdir(path)
    x_train, y_train = [], []
    input_shape = (16, 8, 1)

    for rep in range(1, 10):  # rep 1~9
        for gesture in range(1, 9):
            if subject < 10:
                globals()['data_S{}_G{}_R{}'.format(subject, gesture, rep)] = \
                scipy.io.loadmat(path + file_list[subject - 1] +
                                 '/00{}-00{}-00{}.mat'.format(subject, gesture, rep))['data']
            elif subject >= 10:
                globals()['data_S{}_G{}_R{}'.format(subject, gesture, rep)] = \
                scipy.io.loadmat(path + file_list[subject - 1] +
                                 '/0{}-00{}-00{}.mat'.format(subject, gesture, rep))['data']
            else:
                print(gesture, rep, "!!!!!!!!!!!!!!!!!!")

            data = globals()['data_S{}_G{}_R{}'.format(subject, gesture, rep)]
            data = data[:, sel_four_ch]
            data = wavelength_feature(data, win_len, win_inc)
            x_train.append(data)
            for i in range(len(data)):
                y_train.append(gesture)

    x_train = np.concatenate(x_train)
    y_train = np.array(y_train) - 1

    return x_train, y_train


def import_dataset_test(subject, rep, path, win_len, win_inc, sel_four_ch, TEST_REP = 10):
    file_list = os.listdir(path)
    input_shape = (16, 8, 1)
    x_test, y_test = [], []

    for gesture in range(1, 9):
        if rep == TEST_REP:  # rep 10
            if subject < 10:
                globals()['data_S{}_G{}_R10'.format(subject, gesture)] = \
                scipy.io.loadmat(path + file_list[subject - 1] +
                                 '/00{}-00{}-010.mat'.format(subject, gesture))['data']
            elif subject >= 10:
                globals()['data_S{}_G{}_R10'.format(subject, gesture)] = \
                scipy.io.loadmat(path + file_list[subject - 1] +
                                 '/0{}-00{}-010.mat'.format(subject, gesture))['data']
            else:
                print(subject, rep, "!!!!!!!!!!!!!!!!!!!!!!!!")

            data = globals()['data_S{}_G{}_R10'.format(subject, gesture)]
            data = data[:, sel_four_ch]
            data = wavelength_feature(data, win_len, win_inc)
            x_test.append(data)
            for i in range(len(data)):
                y_test.append(gesture)

    x_test = np.concatenate(x_test)
    y_test = np.array(y_test) - 1

    return x_test, y_test