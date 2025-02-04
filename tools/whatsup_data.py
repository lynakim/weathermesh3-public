import os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

def analyze_folder_structure(base_path):
    variables = {}
    
    for var in sorted(os.listdir(base_path)):
        if var == 'extra':
            continue
        var_path = os.path.join(base_path, var)
        if not os.path.isdir(var_path):
            continue
        
        year_months = sorted(os.listdir(var_path))
        if not year_months:
            continue
        
        start_year_month = year_months[0]
        end_year_month = year_months[-1]
        
        freq_ranges = []
        current_freq = None
        current_start = None
        
        for year_month in year_months:
            ym_path = os.path.join(var_path, year_month)
            files = os.listdir(ym_path)
            if not files:
                continue
            
            timestamps = sorted([int(f.split('.')[0]) for f in files])
            freq = determine_frequency(timestamps)
            if freq != current_freq:
                if current_freq is not None:
                    freq_ranges.append((current_freq, current_start, year_month))
                current_freq = freq
                current_start = year_month
        
        if current_freq is not None:
            freq_ranges.append((current_freq, current_start, end_year_month))
        
        variables[var] = freq_ranges
    
    return variables

def determine_frequency(timestamps):
    if len(timestamps) < 2:
        return None
    
    diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

    max_diff = max(diffs) 
    
    if max_diff == 3600:  
        return "1h"
    elif max_diff == 10800:  
        return "3h"
    elif max_diff == 21600:  
        return "6h"
    else:
        return f"What am i seeing here? {max_diff} at {timestamps[np.argmax(diffs)]}"

def format_output(variables):
    output = []
    for var, freq_ranges in variables.items():
        var_output = f"Variable: {var}\n"
        for freq, start, end in freq_ranges:
            var_output += f"  {freq} from {start[:4]}-{start[4:]} to {end[:4]}-{end[4:]}\n"
        output.append(var_output)
    return "\n".join(output)

results = analyze_folder_structure("/fast/proc/era5/")
print(format_output(results))
results = analyze_folder_structure("/fast/proc/era5/extra")
print(format_output(results))
