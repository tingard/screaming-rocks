import os
from datetime import datetime
import numpy as np
import pandas as pd
from screaming_rocks import SENSOR_IDS
import argparse

parser = argparse.ArgumentParser(
    description=(
        ' Takes the NASA-Sloan Atlas in FITS format, extracts certain columns'
        'and saves as a pandas-friendly pickle file.'
    )
)
parser.add_argument('--data_folder', metavar='/path/to/files/', required=True,
                    type=str, help='Location of input metadata files')
parser.add_argument('--output', metavar='metadata.csv', default='metadata.csv',
                    type=str, help='Desired output csv location')

args = parser.parse_args()


data_folder = args.data_folder
meta = {}
ftpl = 'rct-uop-{:06d}.data.{:05d}.wve'
parse_funcs = {
    'Number of channels': int,
    'Volt range (V)': float,
    'Sampling rate (Hz)': int,
    'Waveform Format': int,
    'Bit Range': int,
    'End Byte': np.int64,
    'Start DateTime': lambda v: datetime.strptime(v, "%d%m%Y %H%M%S.%f"),
    'End DateTime': lambda v: datetime.strptime(v, "%d%m%Y %H%M%S.%f"),
}


def nullfunc(v):
    return v


for sensor_id in SENSOR_IDS:
    for minute in range(40):
        meta_file_loc = os.path.join(
            data_folder,
            ftpl.format(sensor_id, minute)
        )
        if not os.path.isfile(meta_file_loc):
            continue
        with open(meta_file_loc, 'r') as f:
            lines = f.readlines()
            meta[(sensor_id, minute)] = {
                i: parse_funcs.get(i, nullfunc)(j)
                for i, j in (list(map(str.strip, i.split(':'))) for i in lines)
            }

            keys = list(meta.keys())

idx = pd.MultiIndex.from_tuples(keys, names=('sensor_id', 'minute'))
values = [meta[k] for k in keys]
df = pd.Series(values, index=idx).apply(pd.Series)
df.to_csv(args.output)
