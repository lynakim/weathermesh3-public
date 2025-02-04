import time
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
target="/fast/ignored/mars_data/oldoutput_%s.grib"

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
target="/fast/ignored/mars_data/oldoutput_%s.grib"

"""

# req = template % (date_only, hour, datestr, date_only, hour, datestr)

request = ""

# make request for all of 2023 every 6 hours
start_date = datetime(2021, 3, 1)
end_date = datetime(2023, 1, 1)
current_date = start_date
nn = 0
chonks = 20
mm = 0
cmds = ""
while current_date <= end_date:
    #for h in ["00", "06", "12", "18"]:
    #for h in ["06", "18"]:
    for h in ["00", "12"]:
        date_only = current_date.strftime("%Y-%m-%d")
        hour = h + ":00:00"
        datestr = date_only.replace("-", '') + h
        ohp = template % (date_only, hour, datestr, date_only, hour, datestr)
        # only analysis is available for hours 6 and 18
        if h == "06" or h == "18":
            ohp = ohp.replace("stream=oper", "stream=scda")

        request += ohp 
        nn += 1
        if nn > chonks or (current_date == end_date and h == "12"):
            ff = "/fast/ignored/mars_data/oldreq_ch_%d" % mm
            with open(ff, "w") as f:
                f.write(request)
            mm += 1
            cmds += "mars "+ff+"\n"
            nn = 0 
            request = ""
    current_date += timedelta(days=1)

with open("joankcmds4", "w") as f:
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
