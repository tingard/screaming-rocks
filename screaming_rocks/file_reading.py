import os
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
try:
    import dask.dataframe as dd
    from dask.delayed import delayed
except ImportError:
    pass
from . import RATE
from .exceptions import EndOfFileError


FILE_SIZE = 4781506560


def iterate_over_data(sensor_id, start_time=0, batch_size=RATE,
                      data_folder='./'):
    # returns an iterator that can be used to efficiently (and sequentially)
    # read batches of data
    fname = 'rct-uop-{:06d}.data.{:05d}.srm'
    start_byte = np.int64(start_time * RATE * 4 * (16 / 8))
    f_n = 1
    d = np.array([]).reshape(-1, 4)
    while start_byte > FILE_SIZE:
        f_n += 1
        start_byte -= FILE_SIZE
    openf = open(
        os.path.join(data_folder, fname.format(sensor_id, f_n)),
        'rb'
    )
    try:
        while True:
            if start_byte > FILE_SIZE:
                f_n += 1
                start_byte -= FILE_SIZE
                openf.close()
                openf = open(
                    os.path.join(data_folder, fname.format(sensor_id, f_n)),
                    'rb'
                )
                continue
            d = np.fromfile(
                openf,
                dtype=np.uint16,
                count=int(batch_size * 4)
            )
            d.shape = (-1, 4)
            while len(d) < batch_size:
                # we reached the end of the file, switch to the next
                # one and make up the difference
                openf.close()
                f_n += 1
                openf = open(
                    os.path.join(data_folder, fname.format(sensor_id, f_n)),
                    'rb'
                )
                d2 = np.fromfile(
                    openf,
                    dtype=np.uint16,
                    count=int((batch_size - len(d))*4)
                )
                d2.shape = (-1, 4)
                d = np.concatenate((d, d2), axis=0)
                del d2
            yield d
    except Exception as e:
        openf.close()
        raise e


def iterate_to_pandas(sensor_id, start_time, batch_size=RATE,
                      data_folder='./'):
    iterator = iterate_over_data(
        sensor_id,
        start_time=start_time,
        batch_size=batch_size,
        data_folder=data_folder
    )
    idx = (
        pd.TimedeltaIndex(np.arange(batch_size) * 100, unit='ns')
        + pd.Timedelta('{}s'.format(start_time))
    )
    cols = pd.MultiIndex.from_product([(sensor_id,), range(4)])
    while True:
        yield pd.DataFrame(next(iterator), columns=cols, index=idx)
        start_time += batch_size / RATE
        idx = (
            pd.TimedeltaIndex(np.arange(batch_size) * 100, unit='ns')
            + pd.Timedelta('{}s'.format(start_time))
        )


def read_batch(sensor_id, start_time, dt, data_folder=''):
    file_template = 'rct-uop-{:06d}.data.{:05d}.srm'
    start_byte = np.int64(start_time * RATE * int(16 / 8) * 4)
    count = int(dt * RATE * 4)
    i = 1
    while True:
        if start_byte >= FILE_SIZE:
            start_byte -= FILE_SIZE
            i += 1
            continue
        in_file_loc = os.path.join(
            data_folder,
            file_template.format(sensor_id, i)
        )
        with open(in_file_loc):
            d = np.fromfile(in_file_loc, dtype=np.uint16,
                            count=count, offset=start_byte)
            d.shape = (-1, 4)
        if d.shape[0] == count / 4:
            return d
        time_read = d.shape[0] / RATE
        next_file_loc = os.path.join(
            data_folder,
            file_template.format(sensor_id, i + 1)
        )
        if not os.path.isfile(next_file_loc):
            raise EndOfFileError('Time length goes beyond end of file')
        d_next_file = read_batch(
            sensor_id, start_time + time_read, dt - time_read,
            data_folder=data_folder
        )
        d_combined = np.concatenate((d, d_next_file), axis=0)
        if d_combined.shape[0] != dt * RATE:
            raise EndOfFileError('Time length goes beyond end of file')
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


def break_to_pq(sensor_id, start_time, dt, outfolder, data_folder=''):
    """Split the files into many smaller parquet files
    """
    os.makedirs(f'smaller_chunks/{sensor_id:06d}/', exist_ok=True)
    with tqdm() as bar:
        start_time = 0
        while True:
            d = read_batch_as_pandas(
                sensor_id,
                start_time,
                1,
                data_folder=data_folder
            )
            if d.shape[0] == 0:
                break
            df = pd.DataFrame(
                d,
                columns=[f'{sensor_id}-{i}' for i in range(4)],
                index=(
                    pd.TimedeltaIndex(np.arange(d.shape[0]) * 100, unit='ns')
                    + pd.Timedelta('{}s'.format(start_time))
                )
            )
            df.to_parquet(
                os.path.join(
                    outfolder,
                    f'{sensor_id:06d}/{start_time}.parquet.snappy'
                )
            )
            start_time += 1
            bar.update(1)
