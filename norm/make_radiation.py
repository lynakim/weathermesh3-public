import warnings
#warnings.filterwarnings("error")
import sys
import os
import time
import traceback
from pysolar.solar import *
import numpy as np
import multiprocessing
from tqdm import tqdm
#np.seterr(all='print')

res = 0.5
res = 0.25
ires = int(round(res/0.25))

os.makedirs('/fast/consts/neoradiation_%d' % ires, exist_ok=True)
os.makedirs('/fast/consts/solarangle_%d' % ires, exist_ok=True)
lons = np.arange(0, 359.99, res)
lats = np.arange(90, -90.01, -res)[:-1]

Lons, Lats = np.meshgrid(lons, lats)

ANG = True

dates = []
d = datetime.datetime(2008,1,1)
y0 = d
while d < datetime.datetime(2009,1,1):
    dates.append(d.replace(tzinfo=datetime.timezone.utc))
    #d += datetime.timedelta(days=1)
    d += datetime.timedelta(hours=1)

def doit(d):
    rads = np.zeros_like(Lons).astype(np.float32)
    angs = np.zeros_like(rads)
    doy = round((d.replace(hour=0) - y0.replace(tzinfo=datetime.timezone.utc)).total_seconds()/86400)
    #if doy % 2 != 1: return
    hr = d.hour
    #print("doing", doy, hr)
    #oo = '/fast/consts/solarangle_%d/%d_%d.npy' % (ires, doy, hr)
    #if os.path.getmtime(oo) < time.time()-3600*6:
    #    print("uhh", doy, hr)
    #    return
    #return
    #if hr % 4 != int(sys.argv[1]): return
    #print("doing", doy)
    for i in range(Lons.shape[0]):
        for j in range(Lons.shape[1]):
            lat = Lats[i,j]
            lon = Lons[i,j]
            #if ANG:
            #    angs[i,j] = get_altitude_fast(lat, lon, d)
            #    continue
            altitude = get_altitude_fast(lat, lon, d)
            angs[i,j] = altitude
            #continue
            if altitude < 0:
                rads[i,j] = 0
            else:
                try: rads[i,j] = radiation.get_radiation_direct(d, altitude)
                except:
                    fmt = traceback.format_exc()
                    if "timezone aware" not in fmt:
                        traceback.print_exc()
                        print("wtf", lat, lon, d, altitude)
                #print(altitude, rads[i,j])

    np.save('/fast/consts/neoradiation_%d/%d_%d.npy' % (ires, doy, hr), rads)
    np.save('/fast/consts/solarangle_%d/%d_%d.npy' % (ires, doy, hr), angs)

#doit(dates[180]);exit()

pool = multiprocessing.Pool(32)
#pool.map(doit, dates)
s = list(tqdm(pool.imap_unordered(doit, dates), total=len(dates)))
