from utils import *
import deep.meshes as meshes
from functools import partial
import torch
from model import *

from data import *

grid = meshes.get_mesh(64, 16, levels_small)
input_grid = meshes.get_mesh(128, 23)

wh_lev = [levels_full.index(x) for x in levels_small]

p = "ignored/grids/%d_to_%d.pickle" % (input_grid.N, grid.N)
if os.path.exists(p):
    with open(p, 'rb') as f:
        wh_pts = pickle.load(f)

#print(wh_lev)
#print(wh_pts)

def subsetfn(date):
    base = "ignored/proc_%d/pr/%04d%02d/" % (grid.n, date.year, date.month) 
    os.makedirs(base, exist_ok=True)
    pr = get_pressure_input(date, grid=input_grid)
    pr = pr[:, wh_pts, :, :]
    pr = pr[:, :, :, wh_lev]

    for i in range(24):
        fn = base + "%d.tmp.npy" % to_unix(date + timedelta(hours=i))
        np.save(fn, pr[i])
        os.rename(fn, fn.replace(".tmp", ""))

def subsetsfc(date):
    base = "ignored/proc_%d/sfc/%04d%02d/" % (grid.n, date.year, date.month) 
    os.makedirs(base, exist_ok=True)
    pr = get_surface_input(date, grid=input_grid)
    pr = pr[:, wh_pts, :]

    for i in range(24):
        fn = base + "%d.tmp.npy" % to_unix(date + timedelta(hours=i))
        np.save(fn, pr[i])
        os.rename(fn, fn.replace(".tmp", ""))



pool = multiprocessing.Pool(12)
dates = train_dates
dates = valid_dates
dates = train_dates + valid_dates
#list(tqdm(pool.imap_unordered(partial(get_surface_input, grid=grid, ret=False), dates), total=len(dates)))
#list(tqdm(pool.imap_unordered(subsetfn, dates), total=len(dates)))
list(tqdm(pool.imap_unordered(subsetsfc, dates), total=len(dates)))
#get_input(datetime(2018, 1, 5), grid)

