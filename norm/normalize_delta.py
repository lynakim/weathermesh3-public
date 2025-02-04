import multiprocessing
import sys
sys.path.append('..')
from utils import *
from pprint import pprint
import random

cfg = NeoDatasetConfig(WEATHERBENCH=1)
globals().update(cfg.__dict__)


#grid = meshes.get_mesh(128, 23)
#grid = meshes.get_mesh(64, 16)
#grid= meshes.LatLonGrid(subsamp=2, levels=levels_small)

#grid= meshes.LatLonGrid(cfg)#subsamp=1, levels=levels_medium)
grid = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
print(grid.n_levels)
print(grid.subsamp)

mode = "surface"
#mode = "pressure"

if mode == "surface":
    varvar = sfc_vars
else:
    varvar = pressure_vars

print("hey pressure vars", pressure_vars, len(pressure_vars))
print("sfc", len(sfc_vars))

DH = 1
DH = 3
DH = 12
DH = 24 

DH = 72

#DH = 18

DH = 3

FILE_PATH = f"{CONSTS_PATH}/normalization_delta_%dh_28.temp.pickle" % DH
print(FILE_PATH)
NUM = 1337
NUM = 150

#assert not WEATHERBENCH

"""
def delta(d):
    assert DH != 24
    h = list(range(0, 24, DH))
    if mode == "pressure":
        p = get_pressure_input(d, grid, hours=h)
    else:
        p = get_surface_input(d, grid, hours=h)[:,:,:,np.newaxis]
        #print(p.shape)
    d = np.diff(p, axis=0).astype(np.float32)
    avg = np.mean(d, axis=(0,1))
    var = np.var(d, axis=(0,1))
    return avg, var
"""

def delta24(d):
    #assert DH == 24
    #h = list(range(0, 24, DH))
    try:
        #print("yo", d, d+timedelta(days=1))
        assert False, "Anuj has changed get_latlon_input possibly irreparably for this usage, sorry in advance"
        p1, s1 = get_latlon_input(d, cfg, grid, use_cache=True, ret=True)
        p2, s2 = get_latlon_input(d+timedelta(hours=DH), cfg, grid, use_cache=True, ret=True)
        #print("yo", p1.shape, p2.shape, s1.shape, s2.shape)
        """
        p1 = p1.reshape(360*720, 6, 14)
        p2 = p2.reshape(360*720, 6, 14)
        s1 = s1.reshape(360*720, 5)
        s2 = s2.reshape(360*720, 5)
        """
        p1 = p1.reshape(720*1440, 5, 28)
        p2 = p2.reshape(720*1440, 5, 28)
        s1 = s1.reshape(720*1440, 4)
        s2 = s2.reshape(720*1440, 4)
        if mode == "pressure":
            pass
        else:
            p1 = s1[:,:,np.newaxis]
            p2 = s2[:,:,np.newaxis]
            #print(p.shape)
    except ValueError:
        import traceback
        traceback.print_exc()
        print("no consecutive dates?", d)
        return None
    p = np.stack([p1,p2])
    d = np.diff(p, axis=0).astype(np.float32)
    avg = np.mean(d, axis=(0,1))
    var = np.var(d, axis=(0,1))
    #print(var)
    return avg, var

#delta(train_dates[0])
#exit()
#pool = multiprocessing.Pool(16)

pool = multiprocessing.Pool(16)
try:
    with open(FILE_PATH, "rb") as f:
        normalization = pickle.load(f)
except:
    normalization = {}
pprint(normalization)
missing = []
#varlist = ["133_q"]

#train_dates = train_dates[:5]

lst = []
#for date in train_dates:#get_dates((datetime(2018,1,1), datetime(2019,1,1))):
#    lst.append((varname, date))
#print(proc(lst[1])); exit()
#tqdm = lambda x, total: x

#train_dates = train_dates[:100]
big_train_dates = get_dates([(datetime(2015, 1,1), datetime(2019, 1,1), timedelta(hours=1))])
train_dates = random.sample(big_train_dates, NUM)
Data = list(tqdm(pool.imap_unordered(delta24, train_dates), total=len(train_dates)))
#missing.extend([x[1] for x in data if x[0] is None])
#print("missing", missing)

Data = [x for x in Data if x is not None and x[0] is not None]
for idx, varname in enumerate(varvar):
    data = [(x[0][idx, :], x[1][idx, :]) for x in Data]
    #print(len(Data), Data[0][0].shape, data[0][0].shape)
    #data = pool.map(proc, lst)
    mean = np.zeros(data[0][0].shape)
    var = np.zeros(data[0][0].shape)
    m = 0
    for newmean, newvar in data:
        newmean *= 0
        oldmean = mean.copy()
        oldvar = var.copy()
        mean = (m/(m+1)) * oldmean + 1/(m+1) * newmean
        var = 1/(m+1) * newvar + m/(m+1) * oldvar + m/((m+1)**2) * (oldmean - newmean)**2
        #print(varname, ["%e" % y for x, y in zip(mean, np.sqrt(var))])
        m += 1
    #normalization[varname] = (mean.filled(np.nan), var.filled(np.nan))
    normalization[varname] = (mean, var)
    pprint(normalization)
    with open(FILE_PATH, "wb") as f:
        pickle.dump(normalization, f)
