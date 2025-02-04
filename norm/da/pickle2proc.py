# %%
#import sys
#sys.path.append("/huge/users/john/")
import pickle
from datetime import timedelta, datetime
from utils import to_unix
import numpy as np
import os
import json
from multiprocessing import Pool
from tqdm import tqdm

start_date = datetime(2010, 1, 1)
end_date = datetime(2025, 1, 1)
IN_VARS = ['typ', 'etime', 'pres', 'gph', 'temp', 'rh', 'ucomp', 'vcomp']
OUT_VARS = ['id','reltime_seconds','pres_pa','lat_deg','lon_deg', 'unnorm_gp', 'unnorm_temp_k', 'unnorm_rh', 'unnorm_ucomp_mps', 'unnorm_vcomp_mps']
OUT_DIR = "/fast/proc/igra/"
os.makedirs(OUT_DIR,exist_ok=True)
with open(f"{OUT_DIR}/meta.json","w") as f:
    json.dump({
        "vars":OUT_VARS,
        "file_time_format":"%Y%m%d%H",
        "comment":"file name is based on the end of the window. reltime is relative to the end of the window, so it is always negative"
        },f,indent=4)
exit()


# %% 

if 0:
    with open("/huge/igra/pick_smol.pickle", "wb") as f:
        pickle.dump(data[:20], f)

# %%

def save_obs(date,data):
    obs = []
    for station in data:
        if station is None:
            continue
        for sounding in station:
            lat,lon,elev,sounding_start = sounding[0]
            sound_dat = sounding[1]
            td = (date - sounding_start).total_seconds()
            if td > 0 and td < 60*60*3: # check if it was within 3 hours
                
                sound_dat[:,1] = np.nan_to_num(sound_dat[:,1])

                if np.any(np.isnan(sound_dat[:,1])):
                    print(f"({lat:.1f},{lon:.1f}) {sounding_start} Nan in time, skipping")
                    continue
                abs_times = to_unix(sounding_start) + sound_dat[:,1].astype(np.int64)
                idx = np.where(np.logical_and.reduce([to_unix(date) - 60*60 < abs_times,abs_times < to_unix(date),sound_dat[:,IN_VARS.index('pres')] > 0]))[0]
                if len(idx) == 0:
                    print(f"({lat:.1f},{lon:.1f}) {sounding_start} No data in window, skipping")
                    continue
                sound_dat = sound_dat[idx]
                N = len(OUT_VARS)
                sound_dat_new = np.zeros((len(idx), N), dtype=np.float32)
                sound_dat_new[:,OUT_VARS.index('id')] = len(obs)
                sound_dat_new[:,OUT_VARS.index('reltime_seconds')] = (abs_times[idx] - to_unix(date)).astype(np.float32) # convert to seconds relative to file cut off, always should be negative
                sound_dat_new[:,OUT_VARS.index('pres')] = sound_dat[:,IN_VARS.index('pres')]
                sound_dat_new[:,OUT_VARS.index('lat')] = lat
                sound_dat_new[:,OUT_VARS.index('lon')] = lon
                sound_dat_new[:,OUT_VARS.index('unnorm_gp')] = sound_dat[:,IN_VARS.index('gph')] * 9.81
                sound_dat_new[:,OUT_VARS.index('unnorm_temp')] = sound_dat[:,IN_VARS.index('temp')]
                sound_dat_new[:,OUT_VARS.index('unnorm_rh')] = sound_dat[:,IN_VARS.index('rh')] / 100.
                sound_dat_new[:,OUT_VARS.index('unnorm_ucomp')] = sound_dat[:,IN_VARS.index('ucomp')]
                sound_dat_new[:,OUT_VARS.index('unnorm_vcomp')] = sound_dat[:,IN_VARS.index('vcomp')]
                print(f"({lat:.1f},{lon:.1f}) {date} {sound_dat_new.shape} Added")
                obs.append(sound_dat_new)
    if len(obs) == 0:
        return
    obs = np.concatenate(obs,axis=0)
    print(f"#### DONE {date} {obs.shape}")
    #print(obs)
    d = f"{OUT_DIR}/{date.strftime('%Y%m')}/"
    os.makedirs(d,exist_ok=True)
    d = d + f"{date.strftime('%Y%m%d%H')}.npy"
    np.save(d,obs)

def iterate_hours(start_dt, end_dt):
    current_dt = start_dt
    while current_dt <= end_dt:
        yield current_dt
        current_dt += timedelta(hours=1)

def get_time_chunks(start_time, end_time, n_chunks):
    """Split unix timestamp range into n_chunks"""
    times = np.arange(start_time, end_time, 3600, dtype=np.int64)
    return np.array_split(times, n_chunks)

def process_chunk(time_range):
    """Process a chunk of timestamps"""

    filep = '/huge/igra/pick{i}.pickle'
    #file = '/huge/igra/pick_smol.pickle'

    data = []
    for i in range(4):
        print(f"Loading data{i+1}...")
        file = filep.format(i=i+1)
        with open(file,'rb') as f:
            data += pickle.load(f)
        print("loaded")

    print("Processing chunk", time_range[0], time_range[-1])
    for i,timestamp in enumerate(time_range): 
        date = datetime.fromtimestamp(timestamp)
        save_obs(date, data)
        if i % 100 == 0:
            print(f"Processed {i} of {len(time_range)}")

if __name__ == '__main__':
    n_processes = 5
    start_time = to_unix(start_date)
    end_time = to_unix(end_date)

    chunks = get_time_chunks(start_time, end_time, n_processes)
    if 1:
        with Pool(n_processes) as pool:
            list(pool.map(process_chunk, chunks))
    #else:
    #    for chunk in chunks:
    #        process_chunk(chunk)
