import numpy as np

def extract_feature_win(self, x):
    len_x = x.shape[1]
    sum_x = np.sum(x, axis=1)
    mean_x = sum_x / len_x
    ssq_x = np.sum(x ** 2, axis=1)
    std_x = np.sqrt((ssq_x - 2 * sum_x * mean_x + len_x * mean_x ** 2) / (len_x - 1))
    diff_x = np.diff(x, axis=1)

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
    mfl = np.log(np.sqrt(np.mean(diff_x ** 2, axis=1)) + 1)
    mpr = np.mean(x > std_x[:, np.newaxis], axis=1)
    mid = x.shape[1] // 2
    mavs = np.mean(np.abs(x[:, mid:]), axis=1) - np.mean(np.abs(x[:, :mid]), axis=1)

    weight = np.ones_like(x)
    weight[:, :int(0.25 * len_x)] = 0.5
    weight[:, int(0.75 * len_x):] = 0.5
    wmab = np.mean(weight * np.abs(x), axis=1)

    return np.stack([zc, ssc, wl, wamp, mab, msq, rms, v3, lgdec, dabs, mfl, mpr, mavs, wmab], axis=1)


import numpy as np
import pywt
from scipy.fftpack import dct
from scipy.signal import hilbert



### Mean, STD 어케 구함 ??

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
