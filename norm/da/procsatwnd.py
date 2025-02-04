from __future__ import print_function
import ncepbufr
from collections import defaultdict
import datetime
import numpy as np
import json
import sys
import os

satname = sys.argv[1]
datatype = sys.argv[2]
date = sys.argv[3]
assert len(date) == 8, "date must be YYYYMMDD"

OUT_DIR = f"/fast/proc/da/{datatype}"
TAR_FILE = "/huge/proc/ncarsat/%s/%s.%s.tar.gz" % (date[:6], satname, date)

OUT_VARS = ['id', 'reltime_hours', 'lat_deg', 'lon_deg', 'satzenith_deg', 'pres_pa', 'unnorm_ucomp_mps', 'unnorm_vcomp_mps']

metadata = {
    "vars": OUT_VARS,
    "file_time_format": "%Y%m%d%H",
    "comment": "file name is based on the end of the window. reltime is relative to the end of the window, so it is always negative. For example, if directory is 2020121007 and we have data at 6:30am UTC, reltime is -0.5 hours. Data for 5:30am UTC exists in 2020121006."
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(f'{OUT_DIR}/meta.json', 'w') as f:
    json.dump(metadata, f, indent=4)

# SATELLITE IDENTIFIER    
# YEAR
# MONTH
# DAY
# HOUR
# MINUTES
# SECONDS
# LATITUDE (HIGH ACCURACY)  
# LONGITUDE (HIGH ACCURACY)
# LATITUDE (COARSE ACCURACY)
# LONGITUDE (COARSE ACCURACY)
# SATELLITE ZENITH ANGLE
sat_data_str = 'SAID YEAR MNTH DAYS HOUR MINU SECO CLATH CLONH CLAT CLON SAZA'
# PRESSURE 
# WIND DIRECTION
# WIND SPEED
data_str = 'PRLC WDIR WSPD'

import tarfile
import tempfile
from contextlib import contextmanager

@contextmanager
def temp_tar(tar_path):
    temp_dir = tempfile.mkdtemp()
    try:
        print("extracting into", temp_dir, tar_path)
        tarfile.open(tar_path).extractall(temp_dir)
        yield temp_dir
    finally:
        import shutil
        shutil.rmtree(temp_dir)

with temp_tar(TAR_FILE) as tmp:
    ls = sorted(os.listdir(tmp))
    if len(ls) == 1:
        ls = [ls[0] + '/' + x for x in os.listdir(tmp+'/'+ls[0])]
    
    total_datapoints = 0
    skipped_datapoints = 0
    reason = np.zeros(len(OUT_VARS), dtype=np.int32)
    for filename in ls:
        if 'satwnd' not in filename: continue
        
        print("Working on filename: ", filename)
        
        bufr = ncepbufr.open(os.path.join(tmp, filename))
        
        #bufr.print_table()
        all_output = defaultdict(list) # Keys are filename, values are lists of data
        #PREVIOUS_FILENAME = None
        no_lat_lon = set()
        while bufr.advance() == 0:
            while bufr.load_subset() == 0:
                sat_data = bufr.read_subset(sat_data_str).squeeze()
                data = bufr.read_subset(data_str).squeeze()
                
                satid = int(sat_data[0])
                yyyymmddhhmmss = '%04i%02i%02i%02i%02i%02i' % tuple(sat_data[1:7])
                lat = sat_data[7]
                lon = sat_data[8]
                if str(lat) == '--' or str(lon) == '--':
                    lat = sat_data[9]
                    lon = sat_data[10]
                sat_zenith = sat_data[11]
                
                pressure = data[0]
                wind_dir = data[1]
                wind_speed = data[2]
                
                time1 = datetime.datetime.strptime(yyyymmddhhmmss[:-4], "%Y%m%d%H") + datetime.timedelta(hours=1)
                time2 = datetime.datetime.strptime(yyyymmddhhmmss, "%Y%m%d%H%M%S")
                folder_date = yyyymmddhhmmss[:6]
                filename = "%d.npz" % (int(yyyymmddhhmmss[:-4]))
                '''
                if PREVIOUS_FILENAME is None: 
                    print("Working on new file: ", filename)
                    PREVIOUS_FILENAME = filename
                if PREVIOUS_FILENAME != filename: 
                    print("Working on new file: ", filename)
                    PREVIOUS_FILENAME = filename
                '''
                reltime = (time2 - time1).total_seconds() / 3600
                
                ucomp = -wind_speed * np.sin(wind_dir * np.pi/180)
                vcomp = -wind_speed * np.cos(wind_dir * np.pi/180)
                
                data_array = np.zeros(len(OUT_VARS), dtype=np.float32) + np.nan
                data_array[OUT_VARS.index('id')] = satid
                data_array[OUT_VARS.index('reltime_hours')] = reltime
                data_array[OUT_VARS.index('lat_deg')] = lat
                data_array[OUT_VARS.index('lon_deg')] = lon
                data_array[OUT_VARS.index('satzenith_deg')] = sat_zenith
                data_array[OUT_VARS.index('pres_pa')] = pressure
                data_array[OUT_VARS.index('unnorm_ucomp_mps')] = ucomp
                data_array[OUT_VARS.index('unnorm_vcomp_mps')] = vcomp
                
                if any(np.isnan(data_array)): 
                    for idx, val in enumerate(data_array):
                        if np.isnan(val):
                            reason[idx] += 1
                            print(f"NaN found in {OUT_VARS[idx]}")
                            print(data_array)
                            break
                    skipped_datapoints += 1
                else: 
                    all_output[(filename, folder_date)].append(data_array)
                total_datapoints += 1
                
        for (filename, folder_date), data in all_output.items():
            base = f"{OUT_DIR}/{folder_date}/"
            os.makedirs(base, exist_ok=True)
            tm = base + filename.replace(".npz", ".tmp.npz")
            np.savez_compressed(tm, x=data)
            os.rename(tm, base+filename)
        bufr.close()
        
    print(f"Total data points processed: {total_datapoints}")
    print(f"Skipped data points: {skipped_datapoints}")
    print(f"Reason for skipping: {reason}")