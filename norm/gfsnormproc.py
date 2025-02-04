import sys
from metpy.units import units
sys.path.append('..')
from utils import *
import pygrib
import metpy.calc

neo = True

pressure_vars = ["129_z", "130_t", "131_u", "132_v", "135_w", "133_q"]

levlev = levels_small
levlev = levels_medium

factors = {x: [[] for _ in levlev] for x in pressure_vars}

def do(f):
    varlist = ["Geopotential", "Temperature", "U component of wind", "V component of wind", "Vertical velocity", "Relative humidity"]
    #varlist = varlist[:1]
    grbs = pygrib.open(f)
    data = {x: [] for x in varlist}
    levs = {x: [] for x in varlist}
    for x in grbs:
        nam = x.name
        nix = (x.validDate - datetime(1970,1,1)).total_seconds()
        date = x.validDate
        if nam == "Geopotential Height":
            nam = "Geopotential"
        if x.typeOfLevel == "isobaricInhPa" and x.level in levlev and nam in varlist:
            levs[nam].append(x.level)


            lev = x.level

            if neo:
                x = x.values[:]
            else:
                x = downsamp(x.values[:][np.newaxis, :, :, np.newaxis], 2)[0]

            if nam == "Temperature":
                last_temp = x.copy()
                last_temp_lev = lev

            if nam == "Geopotential":
                x = x * 9.80665

            if nam == "Relative humidity":
                assert last_temp_lev == lev
                dp = metpy.calc.dewpoint_from_relative_humidity(last_temp * units.degK, x*units.percent)
                spec = metpy.calc.specific_humidity_from_dewpoint(lev * units.hPa, dp)
                x = (spec/units.dimensionless).magnitude
            data[nam].append(x)
    ref = np.load('/fast/proc/proc_ll_%s/%04d%02d/%d.npz' % ("2_14" if not neo else "1_28", date.year, date.month, nix))['pr'].astype(np.float32)

    with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
        norm = pickle.load(f)

    normdelta = pickle.load(open(f'{CONSTS_PATH}/normalization_delta_%dh_%d.pickle' % (24, len(levlev)), 'rb'))
    #print(normdelta)

    for vi, (v1, v2) in enumerate(zip(varlist, pressure_vars)):
        for i, lev in enumerate(levs[v1]):
            wh = levels_full.index(lev)
            a, b = norm[v2][0][wh], np.sqrt(norm[v2][1][wh])
            xp = (data[v1][i] - a)/b
            #print(v1, np.mean(data[v1][i]), np.mean(xp), np.mean(ref[:,:,vi,i]), np.sum(np.isnan(data[v1][i])), np.prod(data[v1][i].shape))
            ij = levlev.index(lev)
            err = xp - ref[:,:,vi,ij]
            dnorm = np.sqrt(normdelta[v2][1][ij])
            err = err/dnorm
            if neo: factors[v2][ij].append(np.nanmean(np.square(err)))
            else: factors[v2][ij].append(np.nanmean(np.abs(err)))
            #print(v1, lev, np.sqrt(np.nanmean(np.square(err))))
            print(v1, lev, np.nanmean((np.abs(err))))
    return

    print(norm)
    

    for x in varlist:
        print(x, len(data[x]), levs[x])

B = '/fast/ignored/gfsnorm/'
ls = os.listdir(B)
ls = sorted([B+x for x in ls])
#do(ls[1])
for x in ls: do(x)

for v in factors:
    for i in range(len(levlev)):
        factors[v][i] = np.mean(factors[v][i])
    factors[v] = np.array(factors[v])

for v in factors:
    for i in range(len(levlev)):
        if np.isnan(factors[v][i]):
            wh = np.where(~np.isnan(factors[v]))
            print(v, i, levlev[i], "nans", np.array(levlev)[np.where(np.isnan(factors[v]))])
            a = np.array(levlev)[wh]
            b = factors[v][wh]
            c = np.interp(levlev[i], a, b)
            factors[v][i] = c

with open(f"{CONSTS_PATH}/variable_weights_%s.pickle" % ("14" if not neo else "28_neo"), "wb") as f:
    pickle.dump(factors, f)

from pprint import pprint
pprint(factors)
