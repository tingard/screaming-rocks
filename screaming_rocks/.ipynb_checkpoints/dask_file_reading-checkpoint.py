import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.delayed import delayed

from . import to_voltage, RATE


def lazy_read(sensor_id, minute, data_folder, batch_size=1):
    """batch size is in units of seconds"""
    file_location = f'rct-uop-{sensor_id:06d}.data.{minute:05d}.srm'
    f_in = os.path.join(data_folder, file_location)
    _batch_size = int(RATE * 4 * batch_size)
    cols = pd.MultiIndex.from_product([(sensor_id,), range(4)])

    def load(batch_number):
        offset = batch_number * batch_size * 16//8
        with open(f_in, 'rb') as data_file:
            V = to_voltage(
                np.fromfile(
                    data_file,
                    dtype=np.uint16,
                    count=batch_size,
                    offset=offset
                ).reshape(-1, 4)
            )
        idx = (
            pd.TimedeltaIndex(np.arange(len(V)) * 100, unit='ns')
            + pd.Timedelta('{}s'.format(60 * minute + second))
        )
        return pd.DataFrame(V, columns=cols, index=idx)
    return load


def get_minute(sensor_id, minute, data_folder):
    reader = lazy_read(sensor_id, minute, data_folder)
    dfs = [delayed(reader)(i) for i in range(60)]
    # dfs
    return dd.from_delayed(dfs)
