import numpy as np
from scipy.signal import find_peaks
import sys

def partBandpassWhiten(data, freq, psd, lowf, highf):
    white = np.zeros(data.shape, dtype=np.float32)
    mask = 1. * (freq > lowf) * (freq < highf)
    for i in range(data.shape[1]):
        dtilde = np.fft.rfft(data[:, i]) / (psd[:, i]**0.5)
        white[:, i] = np.fft.irfft(dtilde*mask)
    return white


def partPeakFind(wh_data, significance):
    peaks = {}
    peaks_info = {}
    for i in range(wh_data.shape[1]):
        chunk_mean = wh_data[:, i].mean()
        power = np.mean(np.square(wh_data[:, i]-chunk_mean).reshape(-1, 100), axis=1)
        std = np.std(power)
        p, p_info = find_peaks(power, prominence=significance*std, distance=1000)
        # We average over 100 bins, this takes the peak locations back to original sort of data
        p = [(peak*100)+50 for peak in p]
        peaks[i] = p
        peaks_info[i] = p_info
    return peaks, peaks_info


# def whiten_bandpass_findpeaks(wh_data, freq, psd, lowf, highf):
#     white = np.zeros(data.shape, dtype=np.float32)
#     mask = 1. * (freq > lowf) * (freq < highf)
#     for i in range(data.shape[1]):
#         dtilde = np.fft.rfft(data[:, i]) / (psd[:, i]**0.5)
#         white[:, i] = np.fft.irfft(dtilde*mask)
#     peaks = {}
#     power = np.zeros(data.shape, dtype=np.float)
#     for i in range(data.shape[1]):
#         power[:, i] = np.convolve(white[:, i]**2, np.ones(100), mode='same')
#         std = np.std(power[:, i])
#         p = find_peaks(power[:, i], prominence=5*std, distance=1000)
#         peaks[i] = p[0]
#     return peaks