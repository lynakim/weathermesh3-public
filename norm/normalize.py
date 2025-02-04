# http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
import sys
import os
# add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import gc
import traceback

#os.makedirs("ignored/normalization", exist_ok=True)
os.makedirs(f"{CONSTS_PATH}/normalization_tmp", exist_ok=True)
norm_path = f"{CONSTS_PATH}/normalization.pickle"

train_dates = fc_norm_dates
train_dates = get_dates([(datetime(2023, 1, 1), datetime(2024, 8, 31))])
# NOTE: surface variables will take a while without updating the tqdm but it took ~1hr to normalize 1 var for 61 dates
#varlist = ['034_sstk_fixed']
varlist = ['136_tcw']
if all([v in core_pressure_vars for v in varlist]): #mode == "pressure":
    N = 37
    T = 24
elif all([v in all_sfc_vars for v in varlist]):  #mode == "surface":
    N = 1
    T = 1
else:
    assert False

lats = np.arange(90, -90.01, -0.25, dtype=np.float32) * np.pi/180
weights = np.ones((T, N, 721, 1440), dtype=np.float32)
weights *= np.cos(lats)[np.newaxis, np.newaxis, :, np.newaxis]

#CACHE = True
CACHE = 0

oweights = weights

def proc(x):
    global oweights
    var_name, date = x
    agg_hours = 1 #get_agg_hours(var_name) # don't bother redoing normalization for agg, just calculate from base var
    fn = get_raw_nc_filename(var_name, date)
    outpath = f"{CONSTS_PATH}/normalization_tmp/"+fn.split("/")[-1]+f".{agg_hours}h.pickle"
    if CACHE and os.path.exists(outpath):
        with open(outpath, "rb") as f:
            return pickle.load(f)
    try:
        agg_hours
        data = get_dataset(fn, agg_hours=agg_hours)
        weights = oweights
        nan_mask = np.isnan(data[0,0])
        if nan_mask.any():
            assert (nan_mask == np.isnan(data[-1,0])).all(), "nan_mask must be same across all instances" # spot check should be good enough
            weights = np.where(nan_mask[np.newaxis, np.newaxis, :, :], 0, weights)
            data = np.nan_to_num(data, nan=0.0)
        if weights.shape[0] != data.shape[0]:
            weights = np.repeat(weights, data.shape[0], axis=0)
        mean = np.average(data, weights=weights, axis=(0, 2, 3))
        var = np.average((data - mean[np.newaxis, :, np.newaxis, np.newaxis])**2, weights=weights, axis=(0, 2, 3))
        del data
        gc.collect()
    except:
        print("Error computing stuff for", x)
        traceback.print_exc()
        return (None, fn)
    if CACHE:
        with open(outpath+".tmp", "wb") as f:
            pickle.dump((mean, var), f)
        os.rename(outpath+".tmp", outpath)
    return (mean, var)
    

try:
    with open(norm_path, "rb") as f:
        normalization = pickle.load(f)
except:
    normalization = {}
missing = []

for varname in varlist:
    if varname in forecast_sfc_vars:
        # Note this shifts train_dates back by half a month
        train_dates = get_semi_monthly(train_dates)
    else:
        train_dates = get_monthly(train_dates)

    print(f'Normalizing {varname} for {len(train_dates)} dates from {train_dates[0]} to {train_dates[-1]}')

    lst = []
    for date in train_dates:#get_dates((datetime(2018,1,1), datetime(2019,1,1))):
        lst.append((varname, date))
    #pool = multiprocessing.Pool(12)
    #data = list(tqdm(pool.imap_unordered(proc, lst), total=len(lst)))
    data = [proc(lst[0])]
    missing.extend([x[1] for x in data if x[0] is None])
    print("missing", missing)

    data = [x for x in data if x[0] is not None]
    #data = pool.map(proc, lst)
    mean = np.zeros(data[0][0].shape)
    var = np.zeros(data[0][0].shape)
    m = 0
    for newmean, newvar in data:
        oldmean = mean.copy()
        oldvar = var.copy()
        mean = (m/(m+1)) * oldmean + 1/(m+1) * newmean
        var = 1/(m+1) * newvar + m/(m+1) * oldvar + m/((m+1)**2) * (oldmean - newmean)**2
        m += 1
    print(varname, [(round(x,2), round(y,2)) for x, y in zip(mean, np.sqrt(var))])
    normalization[varname] = (mean, var)
    with open(norm_path, "wb") as f:
        pickle.dump(normalization, f)
