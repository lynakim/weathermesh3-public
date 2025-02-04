import sys
from metpy.units import units
sys.path.append('..')
from utils import *
import pygrib
import metpy.calc
neo = True

pressure_vars = ["129_z", "130_t", "131_u", "132_v", "135_w", "133_q"]
sfc_vars = ["165_10u", "166_10v", "167_2t", "168_2d", "151_msl"]

if neo:
    with open(f"{CONSTS_PATH}/variable_weights_28_neo.pickle", "rb") as f:
        factors = pickle.load(f)
else:
    with open(f"{CONSTS_PATH}/variable_weights_14.pickle", "rb") as f:
        factors = pickle.load(f)

factors.update({x: [] for x in sfc_vars})

def do(f):
    varlist = ["Geopotential", "Temperature", "U component of wind", "V component of wind", "Vertical velocity", "Relative humidity"]
    #varlist = 
    #varlist = varlist[:1]
    varlist = ["10 metre U wind component", "10 metre V wind component", "2 metre temperature", "2 metre dewpoint temperature", "Pressure reduced to MSL"]
    grbs = pygrib.open(f)
    """
    for x in grbs:
        if x.level != "isobaricInhPa":
            print(x)
    exit()
    """
    data = {x: [] for x in varlist}
    levs = {x: [] for x in varlist}
    for x in grbs:
        nam = x.name
        nix = (x.validDate - datetime(1970,1,1)).total_seconds()
        date = x.validDate
        if nam == "Geopotential Height":
            nam = "Geopotential"
        if nam in varlist:
            levs[nam].append(x.level)
            lev = x.level
            if not neo: x = downsamp(x.values[:][np.newaxis, :, :, np.newaxis], 2)[0]
            else: x = x.values[:]

            data[nam].append(x)
    #ref = np.load('/fast/proc/proc_ll_2_14/%04d%02d/%d.npz' % (date.year, date.month, nix))['sfc'].astype(np.float32)
    ref = np.load('/fast/proc/proc_ll_%s/%04d%02d/%d.npz' % ("2_14" if not neo else "1_28", date.year, date.month, nix))['sfc'].astype(np.float32)

    with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
        norm = pickle.load(f)

    normdelta = pickle.load(open(f'{CONSTS_PATH}/normalization_delta_%dh_%d.pickle' % (24, 14), 'rb'))
    #print(normdelta)
    for x in data: assert len(data[x]) == 1

    for vi, (v1, v2) in enumerate(zip(varlist, sfc_vars)):
        a, b = norm[v2][0][0], np.sqrt(norm[v2][1][0])
        xp = (data[v1][0] - a)/b
        #i = 0; print(v1, np.mean(data[v1][i]), np.mean(xp), np.mean(ref[:,:,vi]), np.sum(np.isnan(data[v1][i])))
        err = xp - ref[:,:,vi]
        dnorm = np.sqrt(normdelta[v2][1][0])
        err = err/dnorm
        if neo: factors[v2].append(np.nanmean(np.square(err)))
        else: factors[v2].append(np.nanmean(np.abs(err)))
        #print(v1, lev, np.sqrt(np.nanmean(np.square(err))))
        print(v1, lev, np.nanmean((np.abs(err))), factors[v2][-1])
    return

    print(norm)
    

    print(levels_small)
    for x in varlist:
        print(x, len(data[x]), levs[x])

B = '/fast/ignored/gfsnorm/'
ls = os.listdir(B)
ls = sorted([B+x for x in ls])
#do(ls[0])
for x in ls: do(x)

for v in sfc_vars:
    factors[v] = np.mean(factors[v])

with open(f"{CONSTS_PATH}/variable_weights_%s.pickle" % ("14" if not neo else "28_neo"), "wb") as f:
    pickle.dump(factors, f)

from pprint import pprint
pprint(factors)
