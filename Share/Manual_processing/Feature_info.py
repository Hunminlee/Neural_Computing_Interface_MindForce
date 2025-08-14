import numpy as np

# Feature_info.feature_names
fs = round(10e6 / 2048)  # 4883 Hz
lower_cutoff, upper_cutoff = 100, 600


#feature_names = ['Zero Crossing (ZC)', 'Slope Sign Changes (SSC)', 'Waveform Length (WL)', 'WAMP', 'Mean Absolute Value (MAV)', 'Mean Square (MS)', 'Root Mean Square (RMS)',
                 #'v-order 3 (V3)', 'log detector (LD)', 'difference absolute standard deviation value (DASDV)', 'maximum fractal length (MFL)', 'myopulse percentage rate (MPR)',
                 #'mean absolute value slope (MAVS)', 'weighted mean absolute (WMS)',
                 #'Cepstrum Coefficient 1', 'Cepstrum Coefficient 2', 'Cepstrum Coefficient 3', 'Cepstrum Coefficient Average', 'DWTC1', 'DWTC2',
                 #'DWTPC1', 'DWTPC2', 'DWTPC3']

feature_names = ['Zero Crossing (ZC)', 'Slope Sign Changes (SSC)', 'Waveform Length (WL)', 'WAMP', 'Mean Absolute Value (MAV)', 'Mean Square (MS)',
                 'Root Mean Square (RMS)', 'v-order 3 (V3)', 'log detector (LD)', 'difference absolute standard deviation value (DASDV)', 'maximum fractal length (MFL)',
                 'myopulse percentage rate (MPR)', 'mean absolute value slope (MAVS)', 'weighted mean absolute (WMS)']


# 14 features
feat_mean_lst = np.array([0.1, 0.1,  2.5,  0.0,  11.0, 229.0, 13.8, -11.0, 9.0,  3.0,  1.5,  0.0,  0.0,  2.8])
feat_std_lst = np.array([0.02, 0.05, 0.65, 0.02, 4.43, 303.9, 6.85, 12.18, 2.87, 0.87, 0.21, 0.04, 6.68, 1.12])

