import sys
sys.path.append('../../')

from utils import *
import multiprocessing as mp
from functools import partial
import fcntl

def process_month(yyyymm, month_dates):
    dates_with_nan = []
    
    for date in month_dates:
        nix = to_unix(date)
        cp = f'/fast/proc/era5/f000/{yyyymm}/%d.npz' % nix
        if not os.path.exists(cp):
            cp = f'/huge/proc/era5/f000/{yyyymm}/%d.npz' % nix
            if not os.path.exists(cp):
                continue
            
        print("Loading ", cp)
        data = np.load(cp)
        if np.isnan(data['pr']).any():
            print("Found NaaaaaaaaaNs in ", cp)
            dates_with_nan.append(date)
    
    # Write results for this month with file locking
    if dates_with_nan:
        # Write to dates_with_nan.txt
        with open('dates_with_nan.txt', 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Acquire exclusive lock
            try:
                for date in dates_with_nan:
                    f.write(f"{date}\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
        
        # Write to num_nan_dates.txt
        with open('num_nan_dates.txt', 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(f"{yyyymm}: {len(dates_with_nan)} dates with NaNs\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def check_nan_dates():
    dates = get_dates([(datetime(1970, 1, 1), datetime(2007, 12, 31), timedelta(hours=1))])
    dates = dates[::-1]
    
    # Group dates by year-month
    dates_by_month = {}
    for date in dates:
        yyyymm = f"{date.year}{date.month:02d}"
        if yyyymm not in dates_by_month:
            dates_by_month[yyyymm] = []
        dates_by_month[yyyymm].append(date)

    # Create a pool of workers
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    with mp.Pool(num_processes) as pool:
        # Process months in parallel
        pool.starmap(process_month, dates_by_month.items())

if __name__ == '__main__':
    check_nan_dates()
