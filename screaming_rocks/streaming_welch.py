from .dask_file_reading import read_batch
from .partWelch import partWelch
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np


def __task(start_time, sensor_id, window_size, fs, data_folder):
    data = read_batch(
        sensor_id,
        start_time,
        window_size,
        data_folder=data_folder
    )
    return partWelch(data, fs)


def streaming_welch(
    sensor_id,
    start_time,
    end_time,
    window_size=0.001,
    fs=1e7,
    cpu_count=4,
    data_folder='',
):
    """MOTION TO RENAME TO SQWELCH WAS OVERRULED"""
    start_times = np.arange(start_time, end_time, window_size / 2)
    counter = 0
    freq = None
    psd_running_sum = None

    with tqdm(total=len(start_times)) as pbar, Pool(cpu_count) as pool:

        def callback(results_in):
            freq_in, psd_in = results_in
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
            pbar.update(1)

        for start_time in start_times:
            pool.apply_async(
                __task,
                args=(start_time, sensor_id, window_size, fs, data_folder),
                callback=callback
            )
        pool.close()
        pool.join()
    psd_avg = psd_running_sum / counter
    return freq, psd_avg
