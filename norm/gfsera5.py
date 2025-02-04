import random
import sys
sys.path.append("..")
from utils import *

print = builtins.print

globals().update(NeoDatasetConfig(WEATHERBENCH=1).__dict__)

with open("/fast/consts/normalization_adapter_28.pickle", "rb") as f:
    Bylev = pickle.load(f)

with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
    norm = pickle.load(f)

#print(norm["129_z"][1].shape)
#exit()

def cmp(d):
  try:
    nix = to_unix(d)
    v = "pr"
    v = "sfc"
    try: era5 = np.load("/fast/proc/neo_1_28/%04d%02d/%d.npz" % (d.year, d.month, nix))[v]
    except:
        print("uhhh era5", d)
        return
    
    #d += timedelta(hours=3)
    try: gfs = np.load("/fast/proc/gfs_f000/%d.npz" % (to_unix(d)))[v]
    except:
        print("uhhh gfs", d)
        return
    #gfs = np.load("/fast/proc/proc_ll_1_28/%04d%02d/%d.npz" % (d.year, d.month, nix))[v][:720,:,[0,1,2,3,5],:]
    #print(era5.shape, gfs.shape)
    #exit()
    era5 = era5.astype(np.float32)
    gfs = gfs.astype(np.float32)
    gfs = gfs[:720]
    era5 = era5[:720]
    bylev = {}

    """
    for i, v in enumerate(pressure_vars):
        delta = era5[:,:,i,:] - gfs[:,:,i,:]
        bylev[v] = np.sqrt(np.mean(np.square(delta), axis=(0,1)))
        continue
        print(v, "raw500", raw[-1][levels_medium.index(500)])
        wh_lev = [levels_full.index(lev) for lev in levels_medium]
        delta *= np.sqrt(norm[v][1])[np.newaxis, np.newaxis, wh_lev]
        bias = np.mean(delta, axis=(0,1))
        rms = np.sqrt(np.mean(np.square(delta), axis=(0,1)))
        #print(bias.shape, rms.shape)
        #for j, lev in enumerate(levels_medium):
        #    print(v, lev, bias[j], rms[j])

        #print(era5.shape, gfs.shape)
    """

    for i, v in enumerate(sfc_vars):
        delta = era5[:,:,i] - gfs[:,:,i]
        #print(era5.shape, i, v, np.mean(delta), np.std(delta))
        bylev[v] = np.sqrt(np.mean(np.square(delta), axis=(0,1)))
        continue


    return bylev
  except:
      print("uhhh")
      traceback.print_exc()
      return

d = get_dates([(datetime(2021,4,1), datetime(2022, 12, 31), timedelta(hours=12))])
random.seed(0)
r = random.sample(d, 200)
print(cmp(datetime(2022, 6, 21, 0))); exit()

import multiprocessing
pool = multiprocessing.Pool(min(len(r), 8))
tots = pool.map(cmp, r)
tots = [x for x in tots if x is not None]

Bylev.update({x: y*0 for x, y in tots[0].items()})

m = 0
for t in tots:
    for v in tots[0].keys():
        oldmean = 0
        newmean = 0
        oldvar = Bylev[v]
        newvar = t[v]
        Bylev[v] = 1/(m+1) * newvar + m/(m+1) * oldvar + m/((m+1)**2) * (oldmean - newmean)**2
        print("hey", v, Bylev[v])

with open("/fast/consts/normalization_adapter_28.pickle", "wb") as f:
    pickle.dump(Bylev, f)
