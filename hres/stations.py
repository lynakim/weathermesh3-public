import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import random
#from data import *
from hres_utils import *

BASE = "/fast/proc/hres_consolidated/consolidated/"
STATIONS_PATH = BASE + "stations.pkl"
VAL_STATIONS_PATH = BASE + "valid_stations.pickle"
TRAIN_STATIONS_PATH = BASE + "train_stations.pickle"

# Generate (or load) a pickle file with the distinct metar stations and their
# coordinates. This is a dictionary from station name to (lat, lon).
"""
if not os.path.exists(STATIONS_PATH) or 1:
    ls = os.listdir(BASE)
    d = datetime(1994,10,1)
    tots = {}
    while d <= datetime(2024, 1, 1):
        f = np.load(BASE + "metar_%04d_%d_%d.npy" % (d.year, d.month, d.day))
        st = set([(x[0], x[2], x[3]) for x in f]) # (station name, lon, lat)
        for a, b, c in st:
            tots[a] = (c, b) # station name -> (lat, lon)
            assert -90 <= tots[a][0] <= 90, tots[a]
            assert -180 <= tots[a][1] <= 180, tots[a]
        print(d, "uniq", len(tots))
        d += timedelta(days=1)

    pickle.dump(tots, open(BASE+"stations.pickle", "wb"))
else:
    tots = pickle.load(open(BASE+"stations.pickle", "rb"))
"""
tots = pickle.load(open(STATIONS_PATH, "rb"))


# Create a val/train split of the stations by carving out
# areas of `VAL_RADIUS_KM` around randomly selected stations.
VAL_RADIUS_KM = 50
random.seed(0)
#ls = set(tots.keys())
ls = list(range(len(tots)))
valid = set()
lats = []
lons = []

def f(a):
    if a[1] is None: return a[0][1]
    if a[0] is None: return a[1][1]
    r = lambda x: len(str(round(x, 4)))
    dec0 = r(a[0][1][0]) + r(a[0][1][1])
    dec1 = r(a[1][1][0]) + r(a[1][1][1])
    if dec0 > dec1:
        return a[0][1]
    else:
        return a[1][1]

tots = [f(x) for x in tots]

pickle.dump(tots, open(BASE+"stations_latlon.pickle", "wb"))

while True:
    s = random.choice(list(ls))
    for o in list(ls):
        #print(tots[s][0], "ohp", tots[o][0])
        d = gps_dist(*tots[s], *tots[o])
        if d < VAL_RADIUS_KM:
            ls.remove(o)
            valid.add(o)
            lats.append(tots[o][0])
            lons.append(tots[o][1])
    if len(valid) >= len(ls) * 0.05: break

# Save out the validation stations (a set of names).
pickle.dump(valid, open(VAL_STATIONS_PATH, "wb"))
pickle.dump(ls, open(TRAIN_STATIONS_PATH, "wb"))

# Save out a plot of the validation station locations.
# this shows up a.windbornesystems.com/beaches/split.png
plt.scatter(lons, lats)
plt.savefig('/fast/public_html/split.png', dpi=300, bbox_inches='tight')
