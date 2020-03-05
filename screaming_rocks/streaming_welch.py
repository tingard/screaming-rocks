from dask_file_reading import read_batch
from multiprocessing import Pool
import progressbar

widgets = [
    'Processing: ',
    progressbar.Percentage(),
    ' ', progressbar.Bar(),
    ' ', progressbar.ETA()
]
pbar = progressbar.ProgressBar(widgets=widgets, max_value=max_pbar)
counter = 0
cpu_count = 4

pool = Pool(cpu_count)
pool.apply_async(
    f1,
    args=(,),
    callback=f2
)
pool.close()
pool.join()
