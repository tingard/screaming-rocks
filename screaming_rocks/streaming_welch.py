from dask_file_reading import read_batch
from partWelch import partWelch
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np


def streaming_welch(
    sensor_id,
    start_time,
    end_time,
    window_size=0.001,
    fs=1e7,
    data_folder=''
):
    start_times = np.arange(start_time, end_time, window_size / 2)
    pbar = tqdm(total=len(start_times))
    counter = 0
    freq = None
    psd_running_sum = None

    def callback(freq_in, psd_in):
        nonlocal counter
        nonlocal pbar
        nonlocal psd_running_sum
        nonlocal freq
        if psd_running_sum is None:
            psd_running_sum = psd_in
            freq = freq_in
        else:
            psd_running_sum += psd_in
        counter += 1
        pbar.update(counter)

    def task(start_time):
        data = read_batch(
            sensor_id,
            start_time,
            window_size,
            data_folder=data_folder
        )
        return partWelch(data, fs)

    cpu_count = 4
    pool = Pool(cpu_count)
    pool.apply_async(
        task,
        args=start_times,
        callback=callback
    )
    pool.close()
    pool.join()
    pbar.close()
    psd_avg = psd_running_sum / counter
    return psd_avg
