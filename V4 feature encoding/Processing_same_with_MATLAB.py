import os

import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../Share")
import config, utils, baseline


from scipy.signal import lfilter, medfilt
from scipy.signal import hilbert
from sklearn.decomposition import PCA

class EMGFeatureExtractor:
    def __init__(self, feat_mean, feat_std, filter_b, filter_a, Norm_bool):
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.filter_b = filter_b
        self.filter_a = filter_a
        self.buffer = None  # to be set externally
        self.normalization = Norm_bool


    def extract_features(self, win_size=600, win_step=120, feat_exclude=60):
        buf = lfilter(self.filter_b, self.filter_a, self.buffer, axis=1)
        nch, len_x = buf.shape
        n_steps = (len_x - win_size) // win_step + 1

        features = np.zeros((nch, 14, n_steps))
        for i in range(n_steps):
            x = buf[:, i*win_step:i*win_step+win_size]
            features[:, :, i] = self.extract_feature_win(x)

        if self.normalization:
            features = (features - self.feat_mean[:, :, np.newaxis]) / self.feat_std[:, :, np.newaxis]

        if features.shape[2] > feat_exclude:
            features = features[:, :, feat_exclude-1:]  # <-- FIXED HERE
        return features


    def extract_tail_features(self, feat_num, win_size=600, win_step=120):
        n_samples = win_size + (feat_num - 1) * win_step
        if self.buffer.shape[1] < n_samples:
            return None

        buf = self.buffer[:, -n_samples:]
        buf = lfilter(self.filter_b, self.filter_a, buf, axis=1)

        nch, len_x = buf.shape
        n_steps = (len_x - win_size) // win_step + 1
        features = np.zeros((nch, 14, n_steps))

        for i in range(n_steps):
            x = buf[:, i*win_step:i*win_step+win_size]
            features[:, :, i] = self.extract_feature_win(x)

        features = (features - self.feat_mean[:, :, np.newaxis]) / self.feat_std[:, :, np.newaxis]
        return features


    def filter_features(self, features):
        reshaped = features.reshape(features.shape[0]*features.shape[1], -1).T
        pca = PCA(n_components=1)
        pcomp = pca.fit_transform(reshaped).squeeze()
        pcomp = (pcomp - np.mean(pcomp)) / np.std(pcomp)
        med = medfilt(pcomp, kernel_size=5)

        if np.mean(med[:40]) > 0:
            med = -med

        # RMS envelope approximation
        window_size = 3
        rms_env = np.sqrt(np.convolve(med**2, np.ones(window_size)/window_size, mode='same'))
        med = (med - rms_env) - 0.5
        return med


    def extract_feature_win(self, x):
        len_x = x.shape[1]
        sum_x = np.sum(x, axis=1)
        mean_x = sum_x / len_x
        ssq_x = np.sum(x**2, axis=1)
        std_x = np.sqrt((ssq_x - 2*sum_x*mean_x + len_x*mean_x**2)/(len_x-1))
        diff_x = np.diff(x, axis=1)

        zc = np.mean(np.sign(x[:,1:]) != np.sign(x[:,:-1]), axis=1)
        ssc = np.mean(np.sign(diff_x[:,1:]) != np.sign(diff_x[:,:-1]), axis=1)
        wl = np.mean(np.abs(diff_x), axis=1)
        wamp = np.mean(np.abs(np.diff(x, axis=1)) > std_x[:, np.newaxis], axis=1)
        mab = np.mean(np.abs(x), axis=1)
        msq = ssq_x / len_x
        rms = np.sqrt(msq)
        v3 = np.cbrt(np.mean(x**3, axis=1))
        lgdec = np.exp(np.mean(np.log(np.abs(x) + 1), axis=1))
        dabs = np.sqrt(np.mean(diff_x**2, axis=1))
        mfl = np.log(np.sqrt(np.mean(diff_x**2, axis=1)) + 1)
        mpr = np.mean(x > std_x[:, np.newaxis], axis=1)
        mid = x.shape[1] // 2
        mavs = np.mean(np.abs(x[:, mid:]), axis=1) - np.mean(np.abs(x[:, :mid]), axis=1)

        weight = np.ones_like(x)
        weight[:, :int(0.25*len_x)] = 0.5
        weight[:, int(0.75*len_x):] = 0.5
        wmab = np.mean(weight * np.abs(x), axis=1)

        return np.stack([zc, ssc, wl, wamp, mab, msq, rms, v3, lgdec, dabs, mfl, mpr, mavs, wmab], axis=1)

from scipy.signal import cheby2

def create_cheby2_bandpass(fs, low_cutoff, high_cutoff, order=4, rs=30):
    nyq = fs / 2
    low = max(1, low_cutoff) / nyq
    high = min(0.99 * fs / 2, high_cutoff) / nyq
    b, a = cheby2(order, rs, [low, high], btype='bandpass')
    return b, a

#filter_b, filter_a = create_cheby2_bandpass(fs, lower_cutoff, upper_cutoff)