import time
from pprint import pprint
import os
from datetime import datetime, timedelta

template = """retrieve,
class=od,
date=%s,
expver=1,
levelist=10/20/30/50/70/100/150/200/250/300/400/500/600/700/800/850/900/925/950/1000,
levtype=pl,
param=129.128/130.128/131/132/133.128,
step=0,
stream=oper,
grid=0.25/0.25,
time=%s,
type=fc,
target="/fast/ignored/mars_data/archive%s_%s.grib"

retrieve,
class=od,
date=%s,
expver=1,
levtype=sfc,
param=151.128/165.128/166.128/167.128,
step=0,
stream=oper,
grid=0.25/0.25,
time=%s,
type=fc,
target="/fast/ignored/mars_data/archive%s_%s.grib"

"""

# req = template % (date_only, hour, datestr, date_only, hour, datestr)

request = ""

# make request for all of 2023 every 6 hours
start_date = datetime(2016, 3, 1)
end_date = datetime(2024, 3, 1)
d = start_date
dates = []
while d <= end_date:
    for h in [0,6]:
        nix = int((d + timedelta(hours=h) - datetime(1970,1,1)).total_seconds())
        if not os.path.exists("/fast/proc/neohres/f000/%04d%02d/%d.npz" % (d.year, d.month, nix)):
            dates.append(d + timedelta(hours=h))
    d += timedelta(days=1)
current_date = start_date

#pprint(dates);exit()
nn = 0
chonks = 30
mm = 0
cmds = []
for current_date in dates:
        d = current_date
        date_only = current_date.strftime("%Y-%m-%d")
        h = d.hour
        hour = "%02d:00:00/%02d:00:00" % (h, h+12)
        h = "%02d" % h
        datestr = date_only.replace("-", '') + h
        ohp = template % (date_only, hour, h, datestr, date_only, hour, h, datestr)
        # only analysis is available for hours 6 and 18
        if h == "06" or h == "18":
            ohp = ohp.replace("stream=oper", "stream=scda")

        request += ohp 
        nn += 1
        if nn > chonks or (current_date == dates[-1]):
            ff = "/fast/ignored/mars_data/neoreq_ch_%d" % mm
            with open(ff, "w") as f:
                f.write(request)
            mm += 1
            cmds.append("mars "+ff)
            nn = 0 
            request = ""

with open("hres_mars_cmds", "w") as f:
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
