import os
import csv
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

date = sys.argv[1]
month_dir = ''.join(date.split('-')[:2])

print(f"Start date: {date}")
LOAD_DIR = f'/huge/proc/windborne/{month_dir}'

OUT_DIR = f'/fast/proc/da/windborne'

csv_header = ["timestamp", "id", "time", "latitude", "longitude", "altitude", "humidity",
        "pressure", "specific_humidity", "speed_u", "speed_v", "temperature", 
        "updated_at", "mission_id", "mission_name"]

# sh_mgpkg = 'specific_humidity_mg/kg'
OUT_VARS = ['id', 'reltime_hours', 'lat_deg', 'lon_deg', 'pres_pa', 'unnorm_ucomp_mps', 'unnorm_vcomp_mps', 'unnorm_temp_k', 'unnorm_rh', 'unnorm_sh_mgpkg']

metadata = {
    "vars": OUT_VARS,
    "file_time_format": "%Y%m%d%H",
    "comment": "file name is based on the end of the window. reltime is relative to the end of the window, so it is always negative. For example, if directory is 2020121007 and we have data at 6:30am UTC, reltime is -0.5 hours. Data for 5:30am UTC exists in 2020121006."
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(f'{OUT_DIR}/meta.json', 'w') as f:
    json.dump(metadata, f, indent=4)

def get_filenames(date):
    filenames = []
    for file_name in os.listdir(LOAD_DIR):
        date_section = file_name.split('_')[2]
        if date_section == date:
            filenames.append(file_name)
    return filenames
        
def calculate_reltime(timestamp):
    # dir is YYYYMM
    # filename is YYYYMMDDHH
    
    curr_time = datetime.fromtimestamp(timestamp)
    next_hour = curr_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    reltime = (curr_time - next_hour).total_seconds() / 3600
    
    filename = next_hour.strftime('%Y%m%d%H')
    
    dir = next_hour.strftime('%Y%m')
    return reltime, dir, filename
        
def load_data(filenames):
    if filenames == []: return None
    data = defaultdict(list) # Keys are dir and filename, values are list of data arrays
    for filename in filenames:
        with open(LOAD_DIR + '/' + filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            assert csv_header == header
            for row in reader:
                data_point = [float(r) if r != '' else np.nan for r in row[:-2]]
                
                timestamp = data_point[csv_header.index("timestamp")]
                id = data_point[csv_header.index("id")]
                lat_deg = data_point[csv_header.index("latitude")]
                lon_deg = data_point[csv_header.index("longitude")]
                pres_hpa = data_point[csv_header.index("pressure")]
                ucomp = data_point[csv_header.index("speed_u")]
                vcomp = data_point[csv_header.index("speed_v")]
                temp_c = data_point[csv_header.index("temperature")]
                rh = data_point[csv_header.index("humidity")]
                sh = data_point[csv_header.index("specific_humidity")]
                
                reltime, dir, filename = calculate_reltime(timestamp)
                pres_pa = pres_hpa * 100
                temp_k = temp_c + 273.15
                
                if np.isnan(pres_pa): continue 
                
                data_array = np.zeros(len(OUT_VARS), dtype=np.float32) + np.nan
                
                data_array[OUT_VARS.index('id')] = id
                data_array[OUT_VARS.index('reltime_hours')] = reltime
                data_array[OUT_VARS.index('lat_deg')] = lat_deg
                data_array[OUT_VARS.index('lon_deg')] = lon_deg
                data_array[OUT_VARS.index('pres_pa')] = pres_pa
                data_array[OUT_VARS.index('unnorm_ucomp_mps')] = ucomp
                data_array[OUT_VARS.index('unnorm_vcomp_mps')] = vcomp
                data_array[OUT_VARS.index('unnorm_temp_k')] = temp_k
                data_array[OUT_VARS.index('unnorm_rh')] = rh
                data_array[OUT_VARS.index('unnorm_sh_mgpkg')] = sh
                
                data[(dir, filename)].append(data_array)
    return data
    
def save_array(array):
    if array == None: 
        print("Not saving cause there is no data")
        return
    for (dir, filename), values in array.items():
        base = f"{OUT_DIR}/{dir}/"
        os.makedirs(base, exist_ok=True)
        file_path = base + filename + '.npz'
        if os.path.exists(file_path):
            existing_data = np.load(file_path)['x']
            combined_values = np.concatenate([existing_data, values])
        else: combined_values = values
        
        tmp_file_path = base + filename + '.tmp.npz'
        np.savez_compressed(tmp_file_path, x=combined_values)
        os.rename(tmp_file_path, file_path)

filenames = get_filenames(date)
array = load_data(filenames)
save_array(array)


print(f"End date: {date}")