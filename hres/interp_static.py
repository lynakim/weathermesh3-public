from hres_utils import *
import pickle
import time
from scipy.interpolate import RegularGridInterpolator

def ohp(x):
    return max(x, -90+1e-5) # fuckin mcmurdo

def coords_valid(lat, lon):
    return -90 <= lat <= 90 and -180 <= lon <= 180

mode = ""
#mode = "_small"

neo = False
neo = True

#stations = pickle.load(open("/fast/proc/metar/stations.pickle", "rb"))
stations = pickle.load(open("/fast/proc/hres_consolidated/consolidated/stations_latlon.pickle", "rb"))
if neo:
    stations = pickle.load(open("/fast/ignored/merged/merged_stations.pickle", "rb"))
for a in stations:
    lat, lon = a
    try: assert -90 <= lat <= 90 and -180 <= lon <= 180
    except:
        print("wtf", lat, lon)
mx = len(stations)

if neo:
    assert max([x[0] for x in stations.values()])+1 == mx
    is_slop = {v[0]: bool(v[1]) for k, v in stations.items()}
    stations = {v[0]: k for k, v in stations.items()}
    stations = [stations[a] for a in range(mx)]
    is_slop = [is_slop[a] for a in range(mx)]
    stations = [a if coords_valid(*a) else (0.,0.) for a in stations]
    #stations = {k: v for k, v in stations.items() if not np.isnan(k[0])}
else:
    pass
    #print(stations[:10])
    #exit()

if mode == "":
    paths = ["/fast/ignored/elevation/mn30.npy", "/fast/ignored/elevation/sd30.npy"]
    paths = [("/fast/ignored/elevation/mn75.npy", "mn75", 0.5, 16, False), ("/fast/ignored/elevation/mn30.npy", "mn30", 2, 16, False)]
    paths += [("/fast/ignored/modis/2020.npy", "modis_mn75", 0.5, 16, True), ("/fast/ignored/modis/2020_small2.npy", "modis_mn30", 2, 16, True)]
elif mode == "_small":
    paths = ["/fast/ignored/elevation/mn30.npy", "/fast/ignored/elevation/sd30.npy"]
    paths = [("/fast/ignored/elevation/mn75.npy", "mn75", 0.5, 8, False), ("/fast/ignored/elevation/mn30.npy", "mn30", 2, 8, False)]
    paths += [("/fast/ignored/modis/2020.npy", "modis_mn75", 0.5, 8, True), ("/fast/ignored/modis/2020_small2.npy", "modis_mn30", 2, 8, True)]

interps = []
idxs = []
t0 = time.time()
#slist = sorted(list(stations.keys()))
slist = list(range(len(stations)))

pickle.dump(stations, open("/fast/ignored/hres/station_latlon"+("_old" if not neo else "")+".pickle", "wb"))

if neo:
    pickle.dump(is_slop, open("/fast/ignored/hres/is_slop.pickle", "wb"))
#exit()

if mode == "_small": pickle.dump(slist, open("/fast/ignored/hres/station_list"+("_old" if not neo else "")+".pickle", "wb"))
else:
    slist2 = pickle.load(open("/fast/ignored/hres/station_list.pickle", "rb"))
    assert slist2 == slist
#exit()

statics = {}

ex = []

for f in ["topography.npy", "soil_type.npy", "land_mask.npy"]:
    arr = np.load("/fast/consts/"+f)
    arr = np.concatenate((arr, arr[:, 0][:,None]), axis=1)
    x = np.arange(90, -90.01, -0.25)
    y = np.arange(0, 360.01, 0.25)
    sint = RegularGridInterpolator((x, y), arr, method="linear" if arr.dtype.kind == 'f' else "nearest")
    ex.append(sint)

Ohps = []
for station in slist:
    lat, lon = stations[station]
    if lon < 0: lon += 360
    #print("hey pt", lat, lon)
    pts, emb = get_points(lat, lon, 20, 8, with_emb=True)
    emb = np.array([(0.,0.)]+emb)

    ohps = []
    for sint in ex:
        try: oh = sint([(lat,lon)]+pts)
        except:
            print(lat, lon)
            import traceback
            traceback.print_exc()
            exit()
        ohps.append(oh)

    ohps = np.array(ohps).T
    ohps = np.concatenate((ohps, emb), axis=1)
    ohps[:, 0] /= 1000. * 9.8
    ohps[:, 1] /= 7.
    ohps[:, 3] /= 20.
    ohps[:, 4] /= 20.
    Ohps.append(ohps)

    interp, idx = get_interp([(ohp(lat), lon)]+pts, 0.25)

    interps.append(interp)
    idxs.append(idx)
interps = np.array(interps).astype(np.float32)
idxs = np.array(idxs).astype(np.int32)
statics["era5"] = np.array(Ohps).astype(np.float32)
print("hey we wanna make it float16 actually", np.mean(statics["era5"], axis=(0,1)), np.std(statics["era5"], axis=(0,1)))
statics["era5"] = statics["era5"].astype(np.float16)
print("stuff took", time.time()-t0)

print("hiya", interps.shape, idxs.shape, idxs.dtype, interps.dtype)

pickle.dump((interps, idxs), open("/fast/ignored/hres/interps%s"%mode + ("_old" if not neo else "")+".pickle", "wb"))

for path, name, ores, oN, onehot in paths:
    arr = np.load(path, mmap_mode='r')

    res = 180/arr.shape[0]
    print("uhh", np.unravel_index(np.argmax(arr), arr.shape), arr.shape, res, arr.max())
    x = np.arange(90, -90, -res)
    y = np.arange(0, 360, res)
    #print(lons)
    #print(lats)

    #sint = RegularGridInterpolator((x, y), arr, method='nearest' if onehot else 'linear')

    """
    print(x.shape, y.shape, arr.shape)
    sint = RegularGridInterpolator((x, y), arr)
    print("everest", sint([27.9881, 86.9250])[0])
    print("mt whitney", sint([(36.5781, -118.2923 + 360)])[0])
    exit()
    """

    static = []

    heyo =  time.time()
    for station in slist:
        lat, lon = stations[station]
        if lon < 0: lon += 360
        #print("hey pt", lat, lon)
        pts, emb = get_points(lat, lon, ores, oN, with_emb=True)

        pts = [(ohp(lat), lon)] + pts
        emb = np.array([(0.,0.)]+emb)
        #emb2 = np.array([((x-ohp(lat)) * 111, (y-lon)*111*np.cos(lat*np.pi/180) ) for x,y in pts])
        #print("hey", emb.shape, emb.dtype, emb2.shape, emb2.dtype)

        if not onehot:
            interp, idxs = get_interp(pts, res)
            gath = arr[idxs[...,0], idxs[...,1]]
            vals = np.sum(gath * interp, axis=1)
            #print("aa", emb.shape, vals.shape)
            hh = np.hstack((vals[:,None], emb))
            cent = hh[0,0].copy()
            hh[:,0] = (hh[:, 0] - cent)*(1./100)
            hh[:, 1:] /= 20.
            hh[0, 0] = cent*(1./1000)
            static.append(hh.astype(np.float16))
        else:
            idxs = get_nn(pts, res)
            gath = arr[idxs[...,0], idxs[...,1]]
            fill = 17
            if lat < -65:
                fill = 15
            gath[gath == -1] = fill
            gath -= 1
            assert gath.min() >= 0
            assert gath.max() < 17
            """
            out = np.zeros((gath.shape[0], 17))
            out[np.arange(gath.shape[0]), gath] = 1
            """
            #print("huh gath", gath.shape, gath.dtype)
            static.append(gath.astype(np.int8))

        """
        dt1 = time.time()-t0
        t0 = time.time()
        #vals2 = np.array([sint([a,b])[0] for a,b in pts])
        vals2 = sint(np.array(pts))
        dt2 = time.time()-t0
        #print(vals)
        #print(vals2)
        print("err", np.sqrt(np.mean(np.square(vals-vals2))))
        print(dt1, dt2)
        #print(pts, "len", len(pts), "lenvals", len(vals))
        #exit()
        """
    static = np.array(static)
    #statics[name] = static.astype(np.float32)
    statics[name] = static
    print("heyo time", time.time()-heyo, static.shape, static.dtype)

for s in statics:
    print("hey", s, statics[s].shape, statics[s].dtype)

pickle.dump(statics, open("/fast/ignored/hres/statics%s"%mode + ( ("_old" if not neo else "") ) +".pickle", "wb"))

