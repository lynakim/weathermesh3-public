import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

d = datetime(2023,6,21,12,0)

ins = "1bamua"
ins = "atms"

lats = []
lons = []
cols = []
for ins, col in [("1bamua", "blue"), ("atms", "red")]:
    for dh in [0]:#range(-1, 2):
        dx = d + timedelta(hours=dh)
        f = np.load("/fast/proc/satobs/%s/%04d%02d/%d.npz" % (ins, dx.year, dx.month, (dx-datetime(1970,1,1)).total_seconds()))['x']
        lats.append(f[:,2] * 180/np.pi)
        lons.append(f[:,3] * 180/np.pi)
        cols.append([col]*int(f.shape[0]))
        print(ins, dh, f.shape[0])
lat = np.concatenate(lats)
lon = np.concatenate(lons)
col = np.concatenate(cols)
plt.scatter(lon, lat, c=col, s=0.1, alpha=0.05)
plt.savefig('obs.png', dpi=300, bbox_inches='tight')