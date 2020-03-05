import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.delayed import delayed
from . import RATE


def read_batch(sensor_id, start_time, dt, data_folder=''):
    file_template = 'rct-uop-{:06d}.data.{:05d}.srm'
    start_byte = np.int64(start_time * RATE * int(16 / 8) * 4)
    count = int(dt * RATE * 4)
    i = 1
    while True:
        in_file_loc = os.path.join(
            data_folder,
            file_template.format(sensor_id, i)
        )
        fsize = os.path.getsize(in_file_loc)
        if start_byte >= fsize:
            start_byte -= fsize
            i += 1
            continue
        with open(in_file_loc):
            d = np.fromfile(in_file_loc, dtype=np.uint16,
                            count=count, offset=start_byte)
            d.shape = (-1, 4)
        if d.shape[0] == count:
            return d
        time_read = d.shape[0] / RATE
        d_next_file = read_batch(
            sensor_id, start_time + time_read, dt - time_read,
            data_folder=data_folder
        )
        d_combined = np.concatenate((d, d_next_file), axis=0)
        return d_combined


def read_batch_as_pandas(sensor_id, start_time, dt, data_folder=''):
    d = read_batch(sensor_id, start_time, dt, data_folder=data_folder)
    idx = (
        pd.TimedeltaIndex(np.arange(d.shape[0]) * 100, unit='ns')
        + pd.Timedelta('{}s'.format(start_time))
    )
    cols = pd.MultiIndex.from_product([(sensor_id,), range(4)])
    return pd.DataFrame(d, columns=cols, index=idx)


def read_as_dask(sensor_id, start_time, n_seconds, batch_size=0.1,
                 data_folder=''):
    n_batches = int(n_seconds / batch_size)
    if n_batches != n_seconds / batch_size:
        raise ValueError('n seconds must be a multiple of batch size')

    def _f(st):
        return read_batch_as_pandas(
            sensor_id, st, batch_size,
            data_folder=data_folder
        )

    dfs = [
        delayed(_f)(start_time + i * batch_size)
        for i in range(int(n_seconds / batch_size))
    ]
    # dfs
    return dd.from_delayed(dfs)
