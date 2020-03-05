import numpy as np
from datetime import datetime
from scipy.signal.windows import hann


class dataLoader(object):

    def __init__(self, meta_path, data_path, chunk_time=0.1):
        self.meta_path = meta_path
        self.data_path = data_path
        self.load_meta()
        self.chunk_time = chunk_time
        self.psd = None
    
    def __len__(self):
        return int(np.ceil(self.duration/self.chunk_time))
    
    def load_meta(self):
        with open(self.meta_path, 'r') as f:
            lines = [l for l in f]
        self.nchannels = int(lines[0].split(' : ')[-1])
        self.volt_range = float(lines[2].split(': ')[-1])
        self.sample_rate = int(lines[3].split(' : ')[-1])
        self.delta_t = 1./self.sample_rate
        self.bits_per = int(lines[6].split(' : ')[-1])
        self.bits = int(lines[7].split(' : ')[-1])
        self.duration = self.bits/4/2/self.sample_rate
    
    def to_volts(self, arr):
        return (((arr/2**self.bits_per)*self.volt_range)-self.volt_range/2.)*2

    def get_data(self, time=None, offset=0.):
        if time is None:
            samples = -1
            noffset = 0
        else:
            samples = int(self.sample_rate*time)*self.nchannels
            noffset = int(offset*self.sample_rate)*self.nchannels*2
        with open(self.data_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16, count=samples, offset=noffset)
        split = np.zeros((len(data)//4, self.nchannels), dtype=np.float32)
        for i in range(self.nchannels):
            split[:, i] += data[i::self.nchannels]
        array = self.to_volts(split)
        return array
    
    def __getitem__(self, idx):
        if 0 > idx >= self.__len__():
            raise IndexError
        elif idx == self.__len__() - 1:
            return self.get_data(time=self.duration - self.chunk_time*idx,
                                 offset=self.chunk_time*idx)
        else:
            return self.get_data(time=self.chunk_time,
                                 offset=self.chunk_time*idx)
    
    def welch(self):
        nper = int(self.sample_rate*self.chunk_time)
        window = hann(nper)
        window = window.reshape((-1, 1))
        nfft = int(np.floor(self.duration/self.chunk_time))*2 - 1
        freq = np.fft.rfftfreq(nper)
        psd = np.zeros((len(freq), self.nchannels), dtype=np.float32)
        for i in range(nfft):
            timeseries = self.get_data(time=self.chunk_time,
                                       offset=self.chunk_time/2.*i)
            timeseries = timeseries*window
            for j in range(self.nchannels):
                asd = np.fft.rfft(timeseries[:, j])
                psd[:, j] += (asd*asd.conj()).real
        psd *= 1./nper/(window**2).sum()/nfft
        self.psd = psd
        return freq, psd
    
    def whitened(self, idx):
        data = self.__getitem__(idx)
        if self.psd is None:
            _ = self.welch(self.chunk_time)
        white = np.zeros(data.shape, dtype=np.float32)
        for i in range(self.nchannels):
            dtilde = np.fft.rfft(data[:, i])/(self.psd[:, i]**0.5)
            white[:, i] = np.fft.irfft(dtilde)
        return white