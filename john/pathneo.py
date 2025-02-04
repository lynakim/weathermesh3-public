import pathlib
import numpy as np
import time 
from datetime import datetime, timedelta
from pandas import date_range

print("looking for dates")



datemin = datetime(2000,1,1)
datemax = datetime(2026,7,15)


def get_available_times(datemin,datemax):
    base = '/fast/proc/era5/f000/'
    yymm = date_range(datemin-timedelta(days=31),datemax+timedelta(days=31),freq='ME').strftime("%Y%m")
    def seek(yymm):
        return np.array(list(map(lambda p: int(p.stem), pathlib.Path(f'{base}/{yymm}').rglob('[0-9]*.npz'))),dtype=np.int64)
    tots = np.sort(np.concat(list(map(seek,yymm))))
    return tots

print(get_available_times(datemin,datemax))

