import numpy as np
from utils import *
from dateutil.parser import parse as parse_date
import sys

mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
d = parse_date(sys.argv[1]).replace(tzinfo=timezone.utc)
nix = to_unix(d)
assert False, "This usage of get_latlon_input is deprecated"
pr, sfc = get_latlon_input(d, mesh)

sfc = sfc.astype(np.float32)
pr = pr.astype(np.float32)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}, edgeitems=30, linewidth=100000,)

for i in range(6):
    print(i, np.mean(pr[:,:,i], axis=(0,1)))
    print(i, np.std(pr[:,:,i], axis=(0,1)))
    print(i, np.max(np.abs(pr[:,:,i]), axis=(0,1)))
    print("---")
print(pr.shape, sfc.shape)

for i in range(5):
    print(i, np.mean(sfc[:,:,i], axis=(0,1)))
    print(i, np.std(sfc[:,:,i], axis=(0,1)))
    print("---")
