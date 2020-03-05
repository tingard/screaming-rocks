import numpy as np
from datetime import datetime
from scipy.signal.windows import hann
from pycbc.types.timeseries import TimeSeries


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
        self.bits = int(lines[6].split(' : ')[-1])
        start_time = datetime.strptime(lines[8].split(' : ')[-1].strip('\n'), '%d%m%Y %H%M%S.%f')
        end_time = datetime.strptime(lines[9].split(' : ')[-1].strip('\n'), '%d%m%Y %H%M%S.%f')
        self.duration = (end_time - start_time).total_seconds() - 0.001
        self.nsamples = int(np.floor(self.duration*self.sample_rate))
    
    def to_volts(self, arr):
        return (((arr/2**self.bits)*self.volt_range)-self.volt_range/2.)*2

    def get_data(self, time=None, offset=0.):
        if time is None:
            samples = -1
            noffset = 0
        else:
            samples = int(self.sample_rate*time)*self.nchannels
            noffset = int(offset*self.sample_rate)*self.nchannels*2
        with open(self.data_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16, count=samples, offset=noffset)
        split = np.zeros((self.nchannels, len(data)//4), dtype=np.float32)
        for i in range(self.nchannels):
            split[i, :] += data[i::self.nchannels]
        array = self.to_volts(split)
        return [TimeSeries(array[i, :], delta_t=self.delta_t,
                           epoch=offset)
                for i in range(self.nchannels)]
    
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
        nfft = int(np.floor(self.duration/self.chunk_time))*2 - 1
        psd = []
        for i in range(nfft):
            timeseries = self.get_data(time=self.chunk_time,
                                       offset=self.chunk_time/2.*i)
            for j in range(self.nchannels):
                t = timeseries[j]*window
                asd = t.to_frequencyseries()
                if i == 0:
                    psd.append(asd*asd.conj())
                else:
                    psd[j] += asd*asd.conj()
        norm = 2*psd[0].delta_f*nper/(window**2).sum()/nfft
        psd = [p*norm for p in psd]
        self.psd = psd
        return psd
    
    def whitened(self, idx):
        data = self.__getitem__(idx)
        if self.psd is None:
            _ = self.welch(self.chunk_time)
        white = []
        for d, p in zip(data, self.psd):
            dtilde = d.to_frequencyseries()/(p**0.5)
            white.append(dtilde.to_timeseries())
        return white