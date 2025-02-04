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

#globals().update(NeoDatasetConfig(WEATHERBENCH=1).__dict__)

varnames = {"Geopotential height": "129_z", "Temperature": "130_t", "U component of wind": "131_u", "V component of wind": "132_v", "Specific humidity": "133_q"}
sfcvars = {"10 metre U wind component": "165_10u", "10 metre V wind component": "166_10v", "2 metre temperature": "167_2t", "Pressure reduced to MSL": "151_msl"}
iv = {v: k for k, v in varnames.items()}
iv.update({v: k for k, v in sfcvars.items()})

levels = levels_joank

pressure_vars = ["129_z", "130_t", "131_u", "132_v", "133_q"]
sfc_vars = ["165_10u", "166_10v", "167_2t", "151_msl"]

with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
    norm = pickle.load(f)

def upscale(x):
    X = np.zeros((721, 1440), dtype=np.float32) + np.nan
    X[::2, ::2] = x
    X[::2, 1::2] = ((x + np.roll(x,-1,axis=1))/2.)
    X[1::2, :] = (X[::2] + np.roll(X[::2], -1, axis=0))[:-1]/2.
    assert np.isnan(X).sum() == 0
    return X

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
                if x.name == "Geopotential height":
                    fac = 9.80665
                prdic[varnames[x.name]].append(upscale(x.values[:].copy().astype(np.float32)*fac))
                whlev[varnames[x.name]].append(int(x.level))
            if x.name in sfcvars:
                sfcdic[sfcvars[x.name]].append(upscale(x.values[:].copy().astype(np.float32)))
    for v in pressure_vars:
        srt = np.argsort(whlev[v])
        whlev[v] = [whlev[v][i] for i in srt]
        prdic[v] = [prdic[v][i] for i in srt]

    pr = []
    sfc = []
    for v in pressure_vars:
        mean, std2 = norm[v]
        #print("uhhh", whlev[v], levels)
        assert whlev[v] == levels
        wh_lev = [levels_full.index(l) for l in levels]
        x = np.array(prdic[v]).transpose(1,2,0)
        x = (x - mean[np.newaxis, np.newaxis, wh_lev])/np.sqrt(std2)[np.newaxis, np.newaxis, wh_lev]
        print(v, np.mean(x), np.std(x), np.max(np.abs(x)), x.shape)
        pr.append(x.astype(np.float16))
    for v in sfc_vars:
        mean, std2 = norm[v]
        x = np.array(sfcdic[v])[0]
        x = (x - mean[np.newaxis, np.newaxis])/np.sqrt(std2)[np.newaxis, np.newaxis]
        print(v, np.mean(x), np.std(x))
        sfc.append(x[0].astype(np.float16))
    #print(np.array(sfc).shape)
    #print(np.array(pr).shape)
    sfc_data = np.array(sfc).transpose(1, 2, 0)
    data = np.array(pr).transpose(1, 2, 0, 3)
    print("shapes", data.shape, sfc_data.shape)
    pathp = path.replace(".npz", ".tmp.npz")
    np.savez(pathp, pr=data, sfc=sfc_data)
    os.rename(pathp, path)

def get(d):
  path = "/fast/proc/gefsctl/f000/%04d%02d/%d.npz" % (d.year, d.month, to_unix(d))
  if os.path.exists(path):
      return False
  os.makedirs(os.path.dirname(path), exist_ok=True)
  try:
    url = "https://noaa-gefs-pds.s3.amazonaws.com/gefs.%04d%02d%02d/%02d/atmos/pgrb2ap5/gec00.t%02dz.pgrb2a.0p50.f000" % (d.year, d.month, d.day, d.hour, d.hour)
    #      https://noaa-gefs-pds.s3.amazonaws.com/gefs.20241107/12/atmos/pgrb2bp5/gec00.t12z.pgrb2b.0p50.f000
    urls = [url, url.replace('pgrb2a', 'pgrb2b')]
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

"""
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
"""

#get(datetime(2023,12,28,0))
#get(datetime(2021,3,21,0))
#exit()
dates = get_dates([(datetime(2022, 1,1), datetime(2023, 8, 31), timedelta(hours=6))])
dates = get_dates([(datetime(2021, 3,23), datetime(2024, 5, 1), timedelta(hours=12))])

pool = multiprocessing.Pool(12)
pool.map(get, dates)

#get(datetime(2022,6,21))
#proc(["gfs.t06z.pgrb2.0p25.f000", "gfs.t06z.pgrb2b.0p25.f000"])
