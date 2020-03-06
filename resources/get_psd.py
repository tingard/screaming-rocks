import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screaming_rocks import SENSOR_IDS
from screaming_rocks.dask_file_reading import read_batch
from screaming_rocks.streaming_welch import streaming_welch

DATA_FOLDER = '/home/tlingard/bucket'
OUTPUT_FOLDER = '/home/tlingard/output'


for sid in SENSOR_IDS:
    freq, psd = streaming_welch(
        sid,
        0,
        5*60,
        window_size=0.1,
        fs=1e7,
        cpu_count=8,
        data_folder=DATA_FOLDER,
    )
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pd.DataFrame(
        psd,
        index=freq,
        columns=pd.MultiIndex.from_product([(sid,), range(4)])
    ).to_csv(os.path.join(OUTPUT_FOLDER, '{}.csv'.format(sid)))
