import time
import os
from datetime import datetime, timedelta

template = """retrieve,
class=od,
date=%s,
expver=1,
levtype=sfc,
param=23.228/34.128/164.128/168.128/169.128/175.128/246.228/247.228,
step=0,
stream=scda,
grid=0.25/0.25,
time=%s,
type=fc,
target="/fast/ignored/mars_data_sfc/extra618_sfc_%s.grib"

"""
#param=176.128,
#param=142.128/143.128/176.128/201.128/202.128,
#  20.3/23.228/34.128/142.128/143.128/146.128/147.128/164.128/168.128/169.128/175.128/176.128/177.128/201.128/202.128/228.128/246.228/247.228/260109,

# req = template % (date_only, hour, datestr, date_only, hour, datestr)

request = ""

# make request for all of 2023 every 6 hours
start_date = datetime(2021, 3, 1)
end_date = datetime(2024, 9, 5)
current_date = start_date
nn = 0
chonks = 240
mm = 0
cmds = ""
while current_date <= end_date:
    #for h in ["00", "06", "12", "18"]:
    #for h in ["06", "18"]:
    if 1:
        date_only = current_date.strftime("%Y-%m-%d")
        #hour = h + ":00:00"
        hour = "06:00:00/18:00:00"
        datestr = date_only.replace("-", '')# + h
        ohp = template % (date_only, hour, datestr)
        # only analysis is available for hours 6 and 18
        #if h == "06" or h == "18":
        #    ohp = ohp.replace("stream=oper", "stream=scda")

        request += ohp 
        nn += 1
        if nn > chonks or (current_date == end_date):# and h == "12"):
            ff = "/fast/ignored/mars_data_sfc/req6_ch_%d" % mm
            with open(ff, "w") as f:
                f.write(request)
            mm += 1
            cmds += "mars "+ff+"\n"
            nn = 0 
            request = ""
    current_date += timedelta(days=1)

with open("joankcmds7", "w") as f:
    f.write(cmds)

exit()
fn = f"/fast/ignored/mars_data/req_hres_{start_date.year}_{start_date.month}_{end_date.month}"
with open(fn, "w") as f:
    f.write(request)
print("wrote", fn)

exit()

# make ctrl+c happy
time.sleep(1)
os.system(f"mars /fast/ignored/mars_data/req_hres_{start_date.year}_{start_date.month}_{end_date.month}")
