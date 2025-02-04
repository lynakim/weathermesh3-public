from datetime import datetime, timedelta
import os

#https://data.rda.ucar.edu/d559000/wy1984/198312/wrf2d_d01_1983-12-01_02:00:00.nc
fmt = "https://data.rda.ucar.edu/d559000/wy%04d/%04d%02d/wrf%dd_d01_%04d-%02d-%02d_%02d:00:00.nc"

d = datetime(1979, 10, 1)
e = datetime(2022, 9, 30)
e = datetime(2020, 1, 1)

dates = []

while d < e:
    dates.append(d)
    d += timedelta(hours=6)

for d in dates[::-1]:
    output_base = "/huge/proc/conus404/%04d%02d/" % (d.year, d.month)
    os.makedirs(output_base, exist_ok=True)
    wd = d.replace(year=d.year+1) if d.month >= 10 else d
    for r in [2,3]:
        u = fmt % (wd.year, d.year, d.month, r, d.year, d.month, d.day, d.hour)
        op = output_base.replace("/huge/proc/conus404/","")+u.split("/")[-1] 
        if not os.path.exists("/huge/proc/conus404/"+op):
            print(u+"\n\tout="+op )
