import time
import os
from pprint import pprint
from datetime import datetime, timedelta

template = """retrieve,
class=od,
date=%s,
expver=1,
levtype=sfc,
param=142.128/143.128/176.128/201.128/202.128,
step=5/6/11/12,
stream=oper,
grid=0.25/0.25,
time=%s,
type=fc,
target="/fast/ignored/mars_data_sfc/Extraprecip_sfc_%s.grib"

"""
#param=176.128,
#param=142.128/143.128/176.128/201.128/202.128,
#  20.3/23.228/34.128/142.128/143.128/146.128/147.128/164.128/168.128/169.128/175.128/176.128/177.128/201.128/202.128/228.128/246.228/247.228/260109,


template = """retrieve,
class=od,
date=%s,
expver=1,
levtype=sfc,
param=34.128/164.128/168.128/246.228/247.228,
step=0,
stream=oper,
grid=0.25/0.25,
time=%s,
type=fc,
target="/fast/ignored/mars_data_sfc/neoxtra_sfc_%s.grib"

"""
#param=176.128,
#param=142.128/143.128/176.128/201.128/202.128,
#  20.3/23.228/34.128/142.128/143.128/146.128/147.128/164.128/168.128/169.128/175.128/176.128/177.128/201.128/202.128/228.128/246.228/247.228/260109,


# req = template % (date_only, hour, datestr, date_only, hour, datestr)

request = ""

# make request for all of 2023 every 6 hours
start_date = datetime(2021, 3, 1)
end_date = datetime(2024, 3, 5)

start_date = datetime(2024, 3, 1)
end_date = datetime(2024, 10, 5)
current_date = start_date

all_dates = []
d = datetime(2016, 1, 1)
#while d <= datetime(2024, 2, 1):
while d <= datetime(2023, 5, 17):
    if d >= datetime(2021, 3, 1) and d.hour % 12 == 0:
        pass # already have that
    else:
        if d.hour in [0,6]: all_dates.append(d)
    d += timedelta(hours=6)

nn = 0
chonks = 140
mm = 0
cmds = []
for current_date in all_dates:
    #for h in ["00", "06", "12", "18"]:
    #for h in ["06", "18"]:
    if 1:
        date_only = current_date.strftime("%Y-%m-%d")
        #hour = h + ":00:00"
        hour = "%02d:00:00/%02d:00:00" % (current_date.hour, current_date.hour + 12)
        datestr = date_only.replace("-", '')# + h
        ohp = template % (date_only, hour, datestr)
        # only analysis is available for hours 6 and 18
        if current_date.hour == 6:
            ohp = ohp.replace("stream=oper", "stream=scda")

        request += ohp 
        nn += 1
        if nn > chonks or (current_date == all_dates[-1]):# and h == "12"):
            ff = "/fast/ignored/mars_data_sfc/neoreq_ch_%d" % mm
            with open(ff, "w") as f:
                f.write(request)
            mm += 1
            cmds.append("mars "+ff)
            nn = 0 
            request = ""
    current_date += timedelta(days=1)

with open("neoreqs_extra", "w") as f:
    f.write('\n'.join(cmds[::-1]))

exit()
fn = f"/fast/ignored/mars_data/req_hres_{start_date.year}_{start_date.month}_{end_date.month}"
with open(fn, "w") as f:
    f.write(request)
print("wrote", fn)

exit()

# make ctrl+c happy
time.sleep(1)
os.system(f"mars /fast/ignored/mars_data/req_hres_{start_date.year}_{start_date.month}_{end_date.month}")
