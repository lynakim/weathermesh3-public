import sys
sys.path.append('..')
from utils import *
import string
import gc
import random
import requests
import pygrib
import multiprocessing

print = builtins.print

globals().update(NeoDatasetConfig(WEATHERBENCH=1).__dict__)

varnames = {"Geopotential Height": "129_z", "Temperature": "130_t", "U component of wind": "131_u", "V component of wind": "132_v", "Specific humidity": "133_q"}
sfcvars = {"10 metre U wind component": "165_10u", "10 metre V wind component": "166_10v", "2 metre temperature": "167_2t", "Pressure reduced to MSL": "151_msl"}
iv = {v: k for k, v in varnames.items()}
iv.update({v: k for k, v in sfcvars.items()})

with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
    norm = pickle.load(f)

def proc(fs, path):
    prdic = {v: [] for k, v in varnames.items()}
    whlev = {v: [] for k, v in varnames.items()}
    sfcdic = {v: [] for k, v in sfcvars.items()}
    for f in fs:
        f = pygrib.open(f)
        for x in f:
            #print(x)
            if x.name in varnames and x.typeOfLevel == "isobaricInhPa" and x.level in levels:
                #print(x.name, x.level)
                fac = 1
                if x.name == "Geopotential Height":
                    fac = 9.80665
                prdic[varnames[x.name]].append(x.values[:].copy().astype(np.float32)*fac)
                whlev[varnames[x.name]].append(int(x.level))
            if x.name in sfcvars:
                sfcdic[sfcvars[x.name]].append(x.values[:].copy().astype(np.float32))
    for v in pressure_vars:
        for oop in [125,175,875]:
            if oop not in whlev[v]:
                a, b, *_ = np.argpartition(np.abs(np.array(whlev[v])-oop), 1)
                whlev[v].append(oop)
                prdic[v].append((prdic[v][a] + prdic[v][b])/2.)
                print(a,b)
                print(oop, "cf", whlev[v][a], whlev[v][b])
        srt = np.argsort(whlev[v])
        whlev[v] = [whlev[v][i] for i in srt]
        prdic[v] = [prdic[v][i] for i in srt]

    pr = []
    sfc = []
    for v in pressure_vars:
        mean, std2 = norm[v]
        if whlev[v] != levels_medium:
            print("uhh")
            for a,b in zip(whlev[v], levels_medium):
                print(a,b)
        assert whlev[v] == levels_medium
        wh_lev = [levels_full.index(l) for l in levels_medium]
        x = np.array(prdic[v]).transpose(1,2,0)
        x = (x - mean[np.newaxis, np.newaxis, wh_lev])/np.sqrt(std2)[np.newaxis, np.newaxis, wh_lev]
        print(v, np.mean(x), np.std(x), np.max(np.abs(x)))
        pr.append(x.astype(np.float16))
    for v in sfc_vars:
        mean, std2 = norm[v]
        x = np.array(sfcdic[v])[0]
        x = (x - mean[np.newaxis, np.newaxis])/np.sqrt(std2)[np.newaxis, np.newaxis]
        print(v, np.mean(x), np.std(x))
        sfc.append(x[0].astype(np.float16))
    print(np.array(sfc).shape)
    print(np.array(pr).shape)
    sfc_data = np.array(sfc).transpose(1, 2, 0)
    data = np.array(pr).transpose(1, 2, 0, 3)
    pathp = path.replace(".npz", ".tmp.npz")
    np.savez(pathp, pr=data, sfc=sfc_data)
    os.rename(pathp, path)

def get(d, fct=0):
  path = "/fast/proc/gfs_rt/%04d%02d%02d/f%03d/%d.npz" % (d.year, d.month, d.day, fct, to_unix(d))
  if os.path.exists(path):
      return False
  os.makedirs(os.path.dirname(path), exist_ok=True)
  try:
    url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.%04d%02d%02d/%02d/atmos/gfs.t%02dz.pgrb2.0p25.f%03d" % (d.year, d.month, d.day, d.hour, d.hour, fct)
    urls = [url, url.replace('.pgrb2.', '.pgrb2b.')]
    fns = []
    for u in urls:
        r = requests.get(u)
        fn = "/tmp/"+''.join(random.choice(string.ascii_lowercase) for _ in range(10))
        ln = len(r.content)
        if r.status_code == 404: return False
        with open(fn, 'wb') as f:
            f.write(r.content)
        fns.append(fn)
    proc(fns, path)
    return True
  except:
      print("Aa?")
      traceback.print_exc()
  finally:
    try:
        for f in fns:
            try: os.remove(f)
            except: pass
    except:
        print("uhh couldn't delete")
        traceback.print_exc()
        pass

while True:
    d = datetime.utcnow()
    d = d - timedelta(microseconds=d.microsecond, minutes=d.minute,seconds=d.second)
    d = d - timedelta(hours=d.hour%6)
    ret = get(d)
    print("trying to get", d, "got", ret)
    if ret:
        time.sleep(3600*3)
    else:
        time.sleep(600)
    gc.collect()

#get(datetime(2023,12,28,0))
get(datetime(2023,12,28,6))

exit()
dates = get_dates([(datetime(2022, 1,1), datetime(2023, 8, 31), timedelta(hours=6))])
dates = get_dates([(datetime(2021, 3,23), datetime(2022, 1, 1), timedelta(hours=6))])

pool = multiprocessing.Pool(12)
pool.map(get, dates)

#get(datetime(2022,6,21))
#proc(["gfs.t06z.pgrb2.0p25.f000", "gfs.t06z.pgrb2b.0p25.f000"])
