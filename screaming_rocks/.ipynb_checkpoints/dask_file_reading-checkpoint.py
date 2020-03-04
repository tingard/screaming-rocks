import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.delayed import delayed

from . import to_voltage

def lazy_read(sensor_id, minute):
    file_location = f'rct-uop-{sensor_id:06d}.data.{minute:05d}.srm'
    f_in = os.path.join(data_folder, file_location)
    batch_size = int(RATE * 4)
    cols = pd.MultiIndex.from_product([(sensor_id,), range(4)])
    def load(second):
        offset = second * batch_size * 16//8
        with open(f_in, 'rb') as data_file:
            V = to_voltage(
                np.fromfile(data_file, dtype=np.uint16, count=batch_size, offset=offset).reshape(-1, 4)
            )
        start_time = pd.Timedelta('{}s'.format(60 * minute + second))
        idx = pd.TimedeltaIndex(np.arange(len(V)) * 100, unit='ns') + start_time
        return pd.DataFrame(V, columns=cols, index=idx)
    return load

def get_minute(sensor_id, minute):
    reader = lazy_read(sensor_id, minute)
    dfs = [delayed(reader)(i) for i in range(60)]
    # dfs
    return dd.from_delayed(dfs)