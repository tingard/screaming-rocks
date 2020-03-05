import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.delayed import delayed

from . import to_voltage, RATE


def read_batch(sensor_id, start_time, dt, data_folder=''):
    file_template = 'rct-uop-{:06d}.data.{:05d}.srm'
    start_byte = np.int64(start_time * RATE * (16 // 8) * 4)
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


def get_minute(sensor_id, minute, data_folder):
    p = lambda start_time: read_batch_as_pandas(sensor_id, start_time, 1, data_folder=data_folder)
    dfs = [delayed(p)(i) for i in range(60)]
    # dfs
    return dd.from_delayed(dfs)
