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

import numpy as np

def extract_feature_win(x):
    """
    x: (C, win_len)  -> channels first
    Returns: (C, n_features)
    """
    len_x = x.shape[1]
    sum_x = np.sum(x, axis=1)
    mean_x = sum_x / len_x
    ssq_x = np.sum(x ** 2, axis=1)
    std_x = np.sqrt((ssq_x - 2 * sum_x * mean_x + len_x * mean_x ** 2) / (len_x - 1))
    diff_x = np.diff(x, axis=1)

    zc = np.mean(np.sign(x[:, 1:]) != np.sign(x[:, :-1]), axis=1)
    ssc = np.mean(np.sign(diff_x[:, 1:]) != np.sign(diff_x[:, :-1]), axis=1)
    wl = np.mean(np.abs(diff_x), axis=1)
    wamp = np.mean(np.abs(diff_x) > std_x[:, np.newaxis], axis=1)
    mab = np.mean(np.abs(x), axis=1)
    msq = ssq_x / len_x
    rms = np.sqrt(msq)
    v3 = np.cbrt(np.mean(x ** 3, axis=1))
    lgdec = np.exp(np.mean(np.log(np.abs(x) + 1), axis=1))
    dabs = np.sqrt(np.mean(diff_x ** 2, axis=1))
    mfl = np.log(np.sqrt(np.mean(diff_x ** 2, axis=1)) + 1)
    mpr = np.mean(x > std_x[:, np.newaxis], axis=1)
    mid = x.shape[1] // 2
    mavs = np.mean(np.abs(x[:, mid:]), axis=1) - np.mean(np.abs(x[:, :mid]), axis=1)

    weight = np.ones_like(x)
    weight[:, :int(0.25 * len_x)] = 0.5
    weight[:, int(0.75 * len_x):] = 0.5
    wmab = np.mean(weight * np.abs(x), axis=1)

    return np.stack(
        [zc, ssc, wl, wamp, mab, msq, rms, v3, lgdec, dabs, mfl, mpr, mavs, wmab],
        axis=1,
    )  # (C, 14)


def feature_extraction(X, win_len, win_inc):
    """
    X: (T, C) -> time x channels
    Returns: (n_windows, C, n_features)
    """
    T, C = X.shape
    n_features = 14
    n_windows = (T - win_len) // win_inc + 1

    feature_set = np.zeros((n_windows, C, n_features))

    for w in range(n_windows):
        start = w * win_inc
        end = start + win_len
        window = X[start:end, :]  # (win_len, C)

        # Transpose so extract_feature_win gets (C, win_len)
        features = extract_feature_win(window.T)  # (C, 14)

        feature_set[w, :, :] = features  # keep channel × feature

    return feature_set


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

def import_dataset(subject, rep, path, win_len, win_inc, sel_four_ch):
    file_list = os.listdir(path)
    x_train, y_train = [], []
    input_shape = (16, 8, 1)

    for gesture in range(1, 9):
        if rep < 10:
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
            #print('before - ', data.shape)
            data = feature_extraction(data, win_len, win_inc)
            #print('after feature - ', data.shape)

            x_train.append(data)
            for i in range(len(data)):
                y_train.append(gesture)

        elif rep == 10:  # rep 10
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
            x_train.append(data)
            for i in range(len(data)):
                y_train.append(gesture)

    x_train = np.concatenate(x_train)
    y_train = np.array(y_train) - 1

    return x_train, y_train


def import_dataset(subject, rep, path, win_len, win_inc, sel_four_ch, feature_selection):
    file_list = os.listdir(path)
    x_train, y_train = [], []
    input_shape = (16, 8, 1)

    for gesture in range(1, 9):
        if rep < 10:
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
            if feature_selection:
                # print('before - ', data.shape)
                data = feature_extraction(data, win_len, win_inc)
            # print('after feature - ', data.shape)

            x_train.append(data)
            for i in range(len(data)):
                y_train.append(gesture)

        elif rep == 10:  # rep 10
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

            if feature_selection:
                data = feature_extraction(data, win_len, win_inc)
            #data = wavelength_feature(data, win_len, win_inc)
            x_train.append(data)
            for i in range(len(data)):
                y_train.append(gesture)

    x_train = np.concatenate(x_train)
    y_train = np.array(y_train) - 1

    return x_train, y_train