import numpy as np
import math

from types import SimpleNamespace

def gps_dist(lat1, lon1, lat2, lon2):
    rads = np.radians
    lon1, lat1, lon2, lat2 = rads(lon1), rads(lat1), rads(lon2), rads(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def get_points(lat, lon, res, N, with_emb=False):
    out = []
    emb = []
    for dy in np.arange(-N+0.5, N)[::-1]:
        for dx in np.arange(-N+0.5, N):
            f = 111.111
            out.append((min(90, max(-90+1e-5, lat + dy*res/f)), (360 + lon + dx*res/(f * np.cos(lat * np.pi/180))) % 360))
            if with_emb:
                emb.append((dy*res, dx*res))
            #print(dx, dy, "dist", gps_dist(lat, lon, *out[-1]))
    if with_emb:
        return out, emb
    return out

def get_nn(pts, gridres):
    out = []
    idxs = []
    for lat, lon in pts:
        lat = int(round((90-lat)/gridres))
        lat = max(0, min(lat, int(round(180/gridres))-1))
        lon = int(round(lon/gridres))%int(round(360/gridres))
        idxs.append((lat, lon))

    return np.array(idxs)


def get_interp(pts, gridres):
    out = []
    idxs = []
    for lat, lon in pts:
        tlat = int(math.floor((90-lat)/gridres))
        blat = min(tlat+1, int(round(180/gridres))-1)
        llon = int(math.floor(lon/gridres))
        rlon = (llon+1)%int(round(360/gridres))

        v1 = np.array([(llon+1)*gridres - lon, lon - llon*gridres])
        v2 = np.array([(90 - blat*gridres)-lat, lat - (90 - tlat*gridres)])
        det = 1/(-gridres * gridres)
        vals = det * np.outer(v1, v2).flatten()
        idx = [(tlat, llon), (blat, llon), (tlat, rlon), (blat, rlon)]
        out.append(vals)
        idxs.append(idx)
    return np.array(out), np.array(idxs)

def get_interps(allpts, res, N, gridres=0.25):
    a = []
    b = []
    for pt in allpts:
        c, d = get_interp(get_points(*pt, res, N), gridres=gridres)
        a.append(c)
        b.append(d)
    return np.array(a), np.array(b)

default_config = SimpleNamespace()
default_config.gpus = '0'
default_config.nope = not True
default_config.optim = 'shampoo'
default_config.clip_gradient_norm = 4.0
default_config.adam = SimpleNamespace()
default_config.adam.betas= (0.9, 0.999)
default_config.adam.weight_decay= 0.003
default_config.lr_sched = SimpleNamespace()
default_config.lr_sched.lr = 0.7e-4
default_config.lr_sched.warmup_end_step = 10_000
default_config.lr_sched.step_offset = 0
default_config.lr_sched.div_factor = 4
default_config.lr_sched.cosine_period = 200_000
default_config.lr_sched.cosine_en = 1
default_config.lr_sched.cosine_bottom = None
default_config.initial_gradscale = 65536.
#default_config.initial_gradscale = 4096.
default_config.save_every = 500.
default_config.use_l2 = False
default_config.HALF = True
#default_config.weights = np.array([3, 3, 3, 1, 1, 1, 1, 1], dtype=np.float32)
default_config.weights = np.array([3.5, 3.5, 3.5, 4, 4, 0.8, 0.2, 3], dtype=np.float32)
#default_config.weights[3:] = 0
