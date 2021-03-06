import numpy as np
from scipy.signal.windows import hann


def partWelch(data, fs):
    window = hann(data.shape[0])
    data = data * window.reshape((-1, 1))
    freq = np.fft.rfftfreq(data.shape[0], d=1.0 / fs)
    psd = np.zeros((len(freq), data.shape[1]), dtype=np.float32)
    for i in range(data.shape[1]):
        asd = np.fft.rfft(data[:, i])
        psd[:, i] += (asd * asd.conj()).real
    psd *= 1.0 / fs / (window**2).sum()
    return freq, psd
