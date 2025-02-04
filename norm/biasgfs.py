import random
import sys
sys.path.append("..")
from utils import *

print = builtins.print

globals().update(NeoDatasetConfig(WEATHERBENCH=1).__dict__)

with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
    norm = pickle.load(f)

#print(norm["129_z"][1].shape)
#exit()

def cmp(d):
  try:
    nix = to_unix(d)
    v = "pr"
    v = "sfc"
    try: Era5 = np.load("/fast/proc/neo_1_28/%04d%02d/%d.npz" % (d.year, d.month, nix))
    except:
        print("uhhh era5", d)
        return

    try: Hres = np.load("/fast/proc/hres/f000/%04d%02d/%d.npz" % (d.year, d.month, nix))
    except:
        print("uhhh hres", d)
        return

    
    byvar = {}
    #d += timedelta(hours=3)
    try: Gfs = np.load("/fast/proc/gfs_f000/%d.npz" % (to_unix(d)))
    except:
        print("uhhh gfs", d)
        return

    wh_hres = [levels_medium.index(l) for l in levels_tiny]

    #gfs = np.load("/fast/proc/proc_ll_1_28/%04d%02d/%d.npz" % (d.year, d.month, nix))[v][:720,:,[0,1,2,3,5],:]
    #print(era5.shape, gfs.shape)
    #exit()
    for k in ["sfc", "pr"]:
        era5 = Era5[k]
        gfs = Gfs[k]
        hres = Hres[k]

        era5 = era5[:720].astype(np.float32)
        gfs = gfs[:720].astype(np.float32)
        hres = hres[:720, :].astype(np.float32)

        delta_gfs = gfs - era5
        #print("delta_gfs", delta_gfs.shape, np.mean(delta_gfs), np.std(delta_gfs))

        if k == "pr":
            era5 = era5[:,:,:, wh_hres]
        delta_hres = hres - era5
        #print("delta_hres", delta_hres.shape, np.mean(delta_hres, axis=(0,1)), np.std(delta_hres))
        byvar[k] = (delta_gfs, delta_hres)
    return byvar["pr"], byvar["sfc"]
  except:
      print("uhhh???")
      traceback.print_exc()
      return

d = get_dates([(datetime(2021,4,1), datetime(2022, 12, 31), timedelta(hours=12))])
random.seed(0)
random.shuffle(d)

n = 0
for i, x in enumerate(d):
    print(i+1, n, x)
    try: (delta_gfs_pr, delta_hres_pr), (delta_gfs_sfc, delta_hres_sfc) = cmp(x)
    except: continue

    n += 1
    if n == 1:
        Delta_gfs_pr = delta_gfs_pr.astype(np.float64)
        Delta_gfs_sfc = delta_gfs_sfc.astype(np.float64)
        Delta_hres_pr = delta_hres_pr.astype(np.float64)
        Delta_hres_sfc = delta_hres_sfc.astype(np.float64)
    else: # numerical stability is my passion but i am NOT bringing out kahan for this
        # x' = ((n-1) * x + y)/n
        # x' = x - x/n + y/n
        # x += (y-x)/n
        Delta_gfs_pr += (delta_gfs_pr - Delta_gfs_pr)/n
        Delta_gfs_sfc += (delta_gfs_sfc - Delta_gfs_sfc)/n
        Delta_hres_pr += (delta_hres_pr - Delta_hres_pr)/n
        Delta_hres_sfc += (delta_hres_sfc - Delta_hres_sfc)/n
    if n % 25 == 0:
        print("saving!", "gfs", np.mean(Delta_gfs_pr), np.mean(Delta_gfs_sfc), "hres", np.mean(Delta_hres_pr), np.mean(Delta_hres_sfc))
        np.savez("/fast/consts/bias_gfs_hres_era5.tmp.npz", gfs_pr=Delta_gfs_pr.astype(np.float32), gfs_sfc=Delta_gfs_sfc.astype(np.float32), hres_pr=Delta_hres_pr.astype(np.float32), hres_sfc=Delta_hres_sfc.astype(np.float32))
        os.rename("/fast/consts/bias_gfs_hres_era5.tmp.npz", "/fast/consts/bias_gfs_hres_era5.npz")
exit()


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
