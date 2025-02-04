from __future__ import print_function
import ncepbufr
from collections import defaultdict
import matplotlib.pyplot as plt
import datetime
import numpy as np
import json
import sys
import os

name = sys.argv[1]
datatype = sys.argv[2]
date = sys.argv[3]
# val = str(sys.argv[4])
assert len(date) == 8, "date must be YYYYMMDD"

OUT_DIR = f"/fast/proc/da/{datatype}"
TAR_FILE = "/huge/proc/ncarsat/%s/%s.%s.tar.gz" % (date[:6], name, date)

OUT_VARS = ['id', 'reltime_hours', 'lat_deg', 'lon_deg', 'pres_pa', 'unnorm_ucomp_mps', 'unnorm_vcomp_mps', 'unnorm_temp_k', 'unnorm_rh']

metadata = {
    "vars": OUT_VARS,
    "file_time_format": "%Y%m%d%H",
    "comment": "file name is based on the end of the window. reltime is relative to the end of the window, so it is always negative. For example, if directory is 2020121007 and we have data at 6:30am UTC, reltime is -0.5 hours. Data for 5:30am UTC exists in 2020121006."
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(f'{OUT_DIR}/meta.json', 'w') as f:
    json.dump(metadata, f, indent=4)

# REPORT IDENTIFIER
# LATITUDE (COARSE ACCURACY)
# LONGITUDE (COARSE ACCURACY) 
info_data_str = 'RPID YEAR MNTH DAYS HOUR CLAT CLON'
# PRESSURE 
# WIND DIRECTION
# WIND SPEED
# TEMPERATURE/DRY BULB TEMPERATURE
# DEW POINT TEMPERATURE
data_str = 'PRLC WDIR WSPD TMDB TMDP'

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
    
    '''
    memory = []
    total_pressure = 0
    total_gained = 0
    total_new_var = 0
    '''
    for filename in ls:
        if datatype not in filename: continue
        
        print("Working on filename: ", filename)
        
        bufr = ncepbufr.open(os.path.join(tmp, filename))
        
        #bufr.print_table()
        
        all_output = defaultdict(list) # Keys are filename, values are lists of data
        while bufr.advance() == 0:
            while bufr.load_subset() == 0:
                
                def convert_to_array(data):
                    if data.size == 1: return [data]
                    return data
                '''
                def calculate_reltime(yyyymmddhh):
                    time1 = datetime.datetime.strptime(yyyymmddhh, "%Y%m%d%H") + datetime.timedelta(hours=1)
                    time2 = datetime.datetime.strptime(yyyymmddhh, "%Y%m%d%H")
                    folder_date = yyyymmddhh[:6]
                    filename = "%d.npz" % (int(yyyymmddhh))
                    reltime = (time2 - time1).total_seconds() / 3600
                    return folder_date, filename, reltime
                '''
                def convert_wind(speed, direction):
                    ucomp = -speed * np.sin(direction * np.pi/180)
                    vcomp = -speed * np.cos(direction * np.pi/180)
                    return ucomp, vcomp
                def temp_to_humidity(drybulb, dewpoint):
                    # https://bmcnoldy.earth.miami.edu/Humidity.html
                    if drybulb is np.ma.masked or dewpoint is np.ma.masked: return np.nan
                    if drybulb == 0 or dewpoint == 0: return np.nan
                    def convert(val): return np.exp((17.625 * val)/(243.04 + val))
                    numerator = convert(dewpoint)
                    denominator = convert(drybulb)
                    return numerator / denominator
                
                info_data = bufr.read_subset(info_data_str).squeeze()
                id = int(info_data[0])
                yyyymmddhh = '%04i%02i%02i%02i' % tuple(info_data[1:5])
                lat = info_data[-2]
                lon = info_data[-1]
                if lat is np.ma.masked or lon is np.ma.masked: continue
                #folder_date, filename, reltime = calculate_reltime(yyyymmddhh)
                folder_date = yyyymmddhh[:6]
                filename = "%d.npz" % (int(yyyymmddhh))
                reltime = 0
                
                working_data = bufr.read_subset(data_str).squeeze()
                pressure_values = convert_to_array(working_data[0])
                for index, pressure in enumerate(pressure_values):
                    if pressure is np.ma.masked: continue
                    total_datapoints += 1
                    
                    wdir = convert_to_array(working_data[1])[index]
                    wspd = convert_to_array(working_data[2])[index]
                    drybulb = convert_to_array(working_data[3])[index]
                    dewpoint = convert_to_array(working_data[4])[index]
                    '''
                    if (wdir is np.ma.masked) or wspd is np.ma.masked or drybulb is np.ma.masked or dewpoint is np.ma.masked: 
                        print("Skipping because of missing data (not pressure)")
                        skipped_datapoints += 1
                        continue
                    '''
                    if (wdir is np.ma.masked) and (wspd is np.ma.masked) and (drybulb is np.ma.masked) and (dewpoint is np.ma.masked):
                        print("No data for this pressure level, skipping...")
                        continue
                    
                    ucomp, vcomp = convert_wind(wspd, wdir)
                    rh = temp_to_humidity(drybulb, dewpoint)
                                        
                    data_array = np.zeros(len(OUT_VARS), dtype=np.float32) + np.nan
                    data_array[OUT_VARS.index('id')] = id
                    data_array[OUT_VARS.index('reltime_hours')] = reltime
                    data_array[OUT_VARS.index('lat_deg')] = lat
                    data_array[OUT_VARS.index('lon_deg')] = lon
                    data_array[OUT_VARS.index('pres_pa')] = pressure
                    data_array[OUT_VARS.index('unnorm_ucomp_mps')] = ucomp
                    data_array[OUT_VARS.index('unnorm_vcomp_mps')] = vcomp
                    data_array[OUT_VARS.index('unnorm_temp_k')] = drybulb
                    data_array[OUT_VARS.index('unnorm_rh')] = rh
                    
                    all_output[(filename, folder_date)].append(data_array)
                
                '''
                new_var = convert_to_array(bufr.read_subset(val).squeeze())
                gained = np.sum([(pres is not np.ma.masked) and (x is not np.ma.masked) for pres, x in zip(pressure, new_var)])
                total_new_var += np.sum([var is not np.ma.masked for var in new_var])
                total_pressure += np.sum([press is not np.ma.masked for press in pressure])
                total_gained += gained
                '''
        for (filename, folder_date), data in all_output.items():
            base = f"{OUT_DIR}/{folder_date}/"
            os.makedirs(base, exist_ok=True)
            tm = base + filename.replace(".npz", ".tmp.npz")
            np.savez_compressed(tm, x=data)
            os.rename(tm, base+filename)
        bufr.close()
    '''
    print("Variable: ", val)
    print("Total pressure: ", total_pressure)
    print("Total lost: ", total_pressure - total_gained, (total_pressure - total_gained) / total_pressure )
    print("Total after: ", total_gained)
    print("Total new var: ", total_new_var)
    print(" ")
    '''
    print(f"Total data points processed: {total_datapoints}")
    print(f"Skipped data points: {skipped_datapoints}")
    print(f"Reason for skipping: {reason}")
    
'''
# Count number of datapoints collected
import numpy as np
import os

file_name = '/fast/proc/satobs/adpupa'
directories = os.listdir(file_name)
directories = [d for d in directories if 'json' not in d]
count = 0
for dir in directories:
    files = os.listdir(file_name + '/' + dir)
    for f in files:
        if f.endswith('.npz'):
            data = np.load(file_name + '/' + dir + '/' + f)
            count += len(data['x'])
print(f"Total count: {count}")
'''