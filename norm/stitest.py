import numpy as np
import os
import sys

b = "/fast/proc/gfs_f000"
b = sys.argv[1]
ls = os.listdir(b)
for f in ls:
    x = np.load(b+"/"+f)
    for a in ["pr", "sfc"]:
        nn = np.sum(np.isnan(x[a]))
        if nn != 0:
            print("!!!! nans", a, f, nn)
