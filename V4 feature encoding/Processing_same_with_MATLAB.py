import os

import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fftpack import dct
from scipy.signal import hilbert


import sys
sys.path.append("../Share")
import config, utils, baseline


from scipy.signal import lfilter, medfilt
from sklearn.decomposition import PCA

class EMGFeatureExtractor:
    def __init__(self, feat_mean, feat_std, filter_b, filter_a, Norm_bool, num_feature_set):
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.filter_b = filter_b
        self.filter_a = filter_a
        self.buffer = None  # to be set externally
        self.normalization = Norm_bool
        self.num_feature_set = num_feature_set


    def extract_features(self, win_size=600, win_step=120, feat_exclude=60):
        buf = lfilter(self.filter_b, self.filter_a, self.buffer, axis=1)
        nch, len_x = buf.shape
        n_steps = (len_x - win_size) // win_step + 1

        features = np.zeros((nch, self.num_feature_set, n_steps))

        for i in range(n_steps):
            x = buf[:, i*win_step:i*win_step+win_size]
            if self.num_feature_set==23:
                features[:, :, i] = self.extract_feature_win_23_feats(x)
            elif self.num_feature_set==14:
                features[:, :, i] = self.extract_feature_win(x)
            else:
                print("num_feature_set should be either 23 or 14")
                break

        if self.normalization:
            if self.num_feature_set == 14:
                features = (features - self.feat_mean[:, :, np.newaxis]) / self.feat_std[:, :, np.newaxis]

            elif self.num_feature_set == 23:
                # features shape: (n_channels, 23, n_windows)

                # 1️⃣ 기존 14개 feature normalization
                features[:, :14, :] = (features[:, :14, :] - self.feat_mean[:, :, np.newaxis]) / self.feat_std[:, :, np.newaxis]

                # 2️⃣ 새로 추가된 9개 feature는 각 채널별로 time-mean/std 계산
                new_feats = features[:, 14:, :]  # shape: (n_channels, 9, n_windows)

                new_mean = np.mean(new_feats, axis=2, keepdims=True)  # shape: (n_channels, 9, 1)
                new_std = np.std(new_feats, axis=2, keepdims=True) + 1e-6

                features[:, 14:, :] = (new_feats - new_mean) / new_std

        if features.shape[2] > feat_exclude:
            features = features[:, :, feat_exclude-1:]  # <-- FIXED HERE

        return features


    def extract_one_feature_at_a_time(self, target_feature_idx,  win_size=600, win_step=120):
        buf = lfilter(self.filter_b, self.filter_a, self.buffer, axis=1)
        nch, len_x = buf.shape
        n_steps = (len_x - win_size) // win_step + 1

        features = np.zeros((nch, self.num_feature_set, n_steps))

        for i in range(n_steps):
            x = buf[:, i*win_step:i*win_step+win_size]
            if self.num_feature_set==23:
                features[:, :, i] = self.extract_feature_win_23_feats(x)
            elif self.num_feature_set==14:
                features[:, :, i] = self.extract_feature_win(x)
            else:
                print("num_feature_set should be either 23 or 14")
                break
        features = features[:, target_feature_idx, :]  #channel(4) / feature 18 / samples

        return features

    #
    def Normalization(self, features, target_feature_idx, feat_exclude=60):
        if self.normalization:
            if self.num_feature_set == 14:
                features = (features - self.feat_mean[target_feature_idx]) / self.feat_std[target_feature_idx]

            elif self.num_feature_set == 23:
                # features shape: (n_channels, 23, n_windows)

                # 1️⃣ 기존 14개 feature normalization
                if target_feature_idx < 14:
                    features[:, target_feature_idx, :] = (features[:, target_feature_idx, :] - self.feat_mean[target_feature_idx]) / self.feat_std[target_feature_idx]
                else:
                    # 2️⃣ 새로 추가된 9개 feature는 각 채널별로 time-mean/std 계산
                    new_feats = features[:, target_feature_idx, :]  # shape: (n_channels, 9, n_windows)

                new_mean = np.mean(new_feats, keepdims=True)  # shape: (n_channels, 9, 1)
                new_std = np.std(new_feats, keepdims=True) + 1e-6

                features[:, target_feature_idx, :] = (new_feats - new_mean) / new_std

        #print(features.shape, feat_exclude)
        if features.shape[-1] > feat_exclude:
            #print("exclusion initiated")
            features = features[:, feat_exclude-1:]  # <-- FIXED HERE

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

    def extract_feature_win_23_feats(self, x):
        len_x = x.shape[1]
        sum_x = np.sum(x, axis=1)
        mean_x = sum_x / len_x
        ssq_x = np.sum(x ** 2, axis=1)
        std_x = np.sqrt((ssq_x - 2 * sum_x * mean_x + len_x * mean_x ** 2) / (len_x - 1))
        diff_x = np.diff(x, axis=1)

        # Time domain features
        zc = np.mean(np.sign(x[:, 1:]) != np.sign(x[:, :-1]), axis=1)
        ssc = np.mean(np.sign(diff_x[:, 1:]) != np.sign(diff_x[:, :-1]), axis=1)
        wl = np.mean(np.abs(diff_x), axis=1)
        wamp = np.mean(np.abs(np.diff(x, axis=1)) > std_x[:, np.newaxis], axis=1)
        mab = np.mean(np.abs(x), axis=1)
        msq = ssq_x / len_x
        rms = np.sqrt(msq)
        v3 = np.cbrt(np.mean(x ** 3, axis=1))
        lgdec = np.exp(np.mean(np.log(np.abs(x) + 1), axis=1))
        dabs = np.sqrt(np.mean(diff_x ** 2, axis=1))
        mfl = np.log(dabs + 1)
        mpr = np.mean(x > std_x[:, np.newaxis], axis=1)
        mid = x.shape[1] // 2
        mavs = np.mean(np.abs(x[:, mid:]), axis=1) - np.mean(np.abs(x[:, :mid]), axis=1)

        weight = np.ones_like(x)
        weight[:, :int(0.25 * len_x)] = 0.5
        weight[:, int(0.75 * len_x):] = 0.5
        wmab = np.mean(weight * np.abs(x), axis=1)

        # Cepstrum features (using DCT of log magnitude spectrum)
        def compute_cepstrum(row):
            spectrum = np.abs(np.fft.fft(row))
            log_spectrum = np.log(spectrum + 1e-8)
            cepstrum = dct(log_spectrum, norm='ortho')
            return cepstrum[:3], np.mean(cepstrum)

        cc1, cc2, cc3, cca = [], [], [], []
        for row in x:
            cc, mean_cc = compute_cepstrum(row)
            cc1.append(cc[0])
            cc2.append(cc[1])
            cc3.append(cc[2])
            cca.append(mean_cc)
        cc1 = np.array(cc1)
        cc2 = np.array(cc2)
        cc3 = np.array(cc3)
        cca = np.array(cca)

        # DWT features (Haar)
        dwtc1, dwtc2 = [], []
        dwtpc1, dwtpc2, dwtpc3 = [], [], []

        for row in x:
            cA, cD = pywt.dwt(row, 'db4')  # level 1
            dwtc1.append(np.mean(cA))
            dwtc2.append(np.mean(cD))

            wp = pywt.WaveletPacket(data=row, wavelet='db4', maxlevel=2)
            dwtpc1.append(np.mean(wp['aa'].data))
            dwtpc2.append(np.mean(wp['ad'].data))
            dwtpc3.append(np.mean(wp['dd'].data))

        dwtc1 = np.array(dwtc1)
        dwtc2 = np.array(dwtc2)
        dwtpc1 = np.array(dwtpc1)
        dwtpc2 = np.array(dwtpc2)
        dwtpc3 = np.array(dwtpc3)

        # Stack all features
        features = np.stack([
            zc, ssc, wl, wamp, mab, msq, rms, v3, lgdec, dabs, mfl, mpr, mavs, wmab,
            cc1, cc2, cc3, cca,
            dwtc1, dwtc2,
            dwtpc1, dwtpc2, dwtpc3
        ], axis=1)

        return features





from scipy.signal import cheby2

def create_cheby2_bandpass(fs, low_cutoff, high_cutoff, order=4, rs=30):
    nyq = fs / 2
    low = max(1, low_cutoff) / nyq
    high = min(0.99 * fs / 2, high_cutoff) / nyq
    b, a = cheby2(order, rs, [low, high], btype='bandpass')
    return b, a

#filter_b, filter_a = create_cheby2_bandpass(fs, lower_cutoff, upper_cutoff)