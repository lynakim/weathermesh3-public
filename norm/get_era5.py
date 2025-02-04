import sys
sys.path.append('..')
from datetime import datetime, timedelta
import os
from utils import *
from datasets import *


#globals().update(DEFAULT_CONFIG.__dict__)

print = builtins.print

# Some documentation: https://www.notion.so/windborne/ERA5-a93f790c0f6a411bbd9d86f25b88c6c5
varlist = core_pressure_vars + core_sfc_vars #['246_100u', '247_100v']
# logtp is the only weird one, download 142_lsp and 143_cp instead
#varlist = ['168_2d'] #'246_100u', '247_100v']#'15_msnswrf', '142_lsp', '143_cp']
dates = get_dates([(datetime(2024, 9, 1), datetime(2024, 9, 3))])

OUTPUT_PATH = '/huge/proc/raw_era5' # raw_era5_test2 also has some files but better to keep it all in this imo
# we used to use'/slow/era5_test'

for var in varlist:
    # reduce to 1 date for each .nc file we need to request 
    if var in forecast_sfc_vars:
        dates = get_semi_monthly(dates)
    elif var in core_pressure_vars:
        pass # for clarity
    else:
        dates = get_monthly(dates)

    for start in dates:
        os.makedirs(f"{OUTPUT_PATH}/%04d%02d" % (start.year, start.month), exist_ok=True)
        
        url = get_raw_nc_filename(var, start, get_url=True)
        fn = url.split("/")[-1]
        outf = "%04d%02d/%s"%(start.year, start.month, fn)
        if not os.path.exists(f"{OUTPUT_PATH}/"+outf):
            print(url+"\n\tout=%s"%outf) #save to file
        else:
            # print error
            print(f"{OUTPUT_PATH}/{outf} already exists")
            #pass#print(outf, os.path.getsize("/slow/era5/"+outf)/1e6)
        #else: print("ignoring", outf)
    #start += timedelta(days=10)

# HOW TO RUN:
# python get_era5.py > arialist
# inspect arialist file to see if variable and dates are correct
# aria2c --dir=/huge/proc/raw_era5 -c -j16 -x16 -i arialist --auto-file-renaming=false
