import numpy as np
# import scipy.signal as sig
# import scipy.io.wavfile as wav
from scipy.signal.windows import hann


def fast_qtransform(x, y, qval, yeval):
    """Connor's qtransform code"""
    q = np.zeros((len(x), len(yeval)), dtype='complex')
    dx = x[1] - x[0]
    num = len(x)
    freqs = np.fft.fftfreq(num, dx)
    yft = np.fft.fft(y)*dx
    for i in range(0, len(yeval)):
        fi = yeval[i]
        shift = int(fi/(freqs[1]-freqs[0]))
        nstart = int((num + num % 2) / 2)
        yfti = np.zeros((len(yft)), dtype='complex')
        yfti[0:nstart-shift] += yft[shift:nstart]
        yfti[nstart:-shift] += yft[nstart+shift:]
        yfti[-shift:] += yft[:shift]
        windown = int(qval/(fi*dx)) - int(qval/(fi*dx)) % 2
        windowy = np.zeros((num), dtype='float')
        windowy[:int(windown/2)] += hann(windown)[-int(windown/2):]
        windowy[int(-windown/2):] += hann(windown)[:int(windown/2)]
        windowy *= 1./np.sum(windowy)
        windowft = np.fft.fft(windowy)*dx
        windowfti = 1j*windowft
        qft = np.fft.ifft(yfti*np.conj(windowft))*(freqs[1]-freqs[0])*num
        qfti = np.fft.ifft(yfti*np.conj(windowfti))*(freqs[1]-freqs[0])*num
        q[:, i] = (
            qft*np.conj(qft) + qfti*np.conj(qfti)
        ) / np.sum(windowft * np.conj(windowft))
    return np.abs(q)
