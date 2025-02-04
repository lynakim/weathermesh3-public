import torch
import time
import socket
import os, threading
import math
from typing import List, Tuple, Any, Dict
import pickle
import time
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from pprint import pprint



def utctimestamp(dt: datetime) -> int:
    return (dt - datetime(1970,1,1)).total_seconds()

BASE = "/fast/proc/hres_consolidated/consolidated/"
BASE = "/huge/ignored/merged/"
STATIONS_PATH = "/huge/ignored/hres/station_latlon.pickle"
ROUNDING_DATA_PATH = "/huge/proc/hres_consolidated/consolidated/station_rounding/"
ERA5_BASE = "/fast/proc/era5/f000/"
ERA5_BASE = "/huge/proc/era5/f000/"
#VALID_STATIONS = BASE + "valid_stations.pickle"

t0 = time.time()
#Elevation30 = np.memmap("/huge/ignored/elevation/mn30.npy", dtype=np.float32, shape=(21600, 43200), mode='r')
Elevation30 = np.load("/huge/ignored/elevation/mn30.npy", mmap_mode='r')
print("took", time.time()-t0, Elevation30.dtype, Elevation30.shape)

t0 = time.time()
#Modis30 = np.memmap("/huge/ignored/modis/2020_small2.npy", dtype=np.int8, shape=(21600, 43200), mode='r')
Modis30 = np.load("/huge/ignored/modis/2020_small2.npy", mmap_mode='r')
print("took", time.time()-t0, Modis30.dtype, Modis30.shape)

lons = np.arange(0, 360, 0.25) * np.pi/180
lats = np.arange(90, -90, -0.25) * np.pi/180
Lons, Lats = np.meshgrid(lons, lats)

cLons = np.cos(Lons)
sLons = np.sin(Lons)
cLats = np.cos(Lats)
sLats = np.sin(Lats)

orography = np.load("/huge/consts/topography.npy")[:720]
orography /= np.max(orography)

Static_sfc = np.stack([orography, cLons, sLons, cLats, sLats], axis=2)

DATA_START = datetime(1979, 1, 2)
class HresDataset(Dataset):
    def __init__(self, batch_size, years=None, variables=None, validation=False, old_fraction=0.9):
        """Validation means load data post 2022 only."""
        print("initializing?")
        self.batch_size = batch_size
        self.old_fraction = old_fraction

        # Each data point consists of all the points in
        # [t - half_width_hours, t + half_width_hours], where t
        # is the time at which we have grid data (every 3 hours).
        self.half_width_hours = 10.1/60
        self.dt = self.half_width_hours * 3600

        # Starting point of all the data.
        start = DATA_START

        if years is None:
            years = list(range(1979,2021))

        self.weights = []
        # Compute all the points we have ERA5 data for.
        if os.path.exists(ERA5_BASE):
            if not validation:
                year_month_dirs = sorted([
                    x for x in os.listdir(ERA5_BASE)
                    if os.path.isdir(ERA5_BASE + x) and 
                    int(x[:4]) >= start.year and
                    int(x[4:]) >= start.month and int(x[:4]) in years])
            else:
                year_month_dirs = sorted([
                    x for x in os.listdir(ERA5_BASE)
                    if os.path.isdir(ERA5_BASE + x) and 
                    int(x[:4]) == 2022 and int(x[4:]) in [1, 7]]) # changed validation to only jan & jul 2022
            self.datetimes = []
            for d in year_month_dirs:
                timestamps = sorted([
                    datetime.utcfromtimestamp(int(x[:-4]))
                    for x in os.listdir(ERA5_BASE + d)
                    if x.endswith('.npz')])
                timestamps = [x for x in timestamps if os.path.exists(BASE + f"{x.year}/{x.month:02}/{x.year}{x.month:02}{x.day:02}{x.hour:02}.npz")]
                ex = [t for t in timestamps if t >= start]
                self.datetimes += ex
                self.weights += [os.path.getsize(BASE + f"{x.year}/{x.month:02}/{x.year}{x.month:02}{x.day:02}{x.hour:02}.npz")/1e6 for x in ex]
        else:
            if validation:
                raise NotImplementedError("validation not implemented for this case oops")
            self.datetimes = []
            print("oooops we're far from home")
            st = datetime(1979,1,1)
            while st <= datetime(2022, 1, 1):
                self.datetimes.append(st)
                st += timedelta(hours=6)

        # Currently, these must all have the same type: np.float32.
        self.variables = variables or ["tmpf", "dwpf", "mslp", '10u', '10v', 'skyl1', 'vsby', 'precip']
        
        self.stations_latlon = pickle.load(open(STATIONS_PATH, "rb"))
        self.nnew = len(self.stations_latlon)
        self.stations_latlon_old = pickle.load(open(STATIONS_PATH.replace("latlon", "latlon_old"), "rb"))
        self.nold = len(self.stations_latlon_old)
        self.is_slop = pickle.load(open(STATIONS_PATH.replace("station_latlon", "is_slop"), "rb"))
        assert type(self.is_slop[0]) == bool
        self.is_slop = np.concatenate([self.is_slop, np.zeros(len(self.stations_latlon_old), dtype=bool)])
        #print("uhh, self" ,self.stations_latlon[:10])
        self.stations_latlon_ext = np.concatenate([self.stations_latlon, self.stations_latlon_old])
        #self.stations_latlon = pickle.load(open(BASE + "stations_latlon.pickle", "rb"))
        #self.valid_stations = pickle.load(open(VALID_STATIONS, "rb")) # List[int]

        with open("/huge/consts/normalization.pickle", "rb") as f:
            self.normalization = pickle.load(f)
        self.normalization['tmpf'] = self.normalization['167_2t']
        self.normalization['dwpf'] = self.normalization['168_2d']
        self.normalization['mslp'] = self.normalization['151_msl']
        self.normalization['10u'] = self.normalization['165_10u']
        self.normalization['10v'] = self.normalization['166_10v']
        self.normalization['vsby'] = (np.array([10]), np.array([5**2]))
        self.normalization['skyl1'] = (np.array([2000]), np.array([4000**2]))
        self.normalization['precip'] = (np.array([10]), np.array([10**2]))
        self.normalizations = [self.normalization[x] for x in self.variables]

        self.COLUMNS = [
            "timestamp",
            "station",
            "mslp",
            "tmpf",
            "dwpf",
            "10u",
            "10v",
            "skyl1",
            "vsby",
            "precip"
        ]

        self.rounding_weights = {}
        for y in range(1979,2024):
            try: self.load_rounding_weights(y)
            except: pass
        #self.load_rounding_weights() # sets self.rounding_weights

    def load_rounding_weights(self, year):
        if year in self.rounding_weights: return self.rounding_weights[year]
        # print("hey uh gotta load", year, threading.get_ident(), os.getpid())

        with open(ROUNDING_DATA_PATH + f"{year}.pkl", "rb") as f:
            aa = np.array(pickle.load(f))
            #print(year, np.isnan(aa).sum(), aa.shape)
            self.rounding_weights[year] = np.concatenate([np.zeros(self.nnew)+0.1, aa])
        return self.rounding_weights[year]

    def __len__(self) -> int:
        return len(self.datetimes)

    @staticmethod
    def load_era5(dt: datetime) -> Dict:
        timestamp = int(utctimestamp(dt))
        path = f"{ERA5_BASE}{dt.year}{dt.month:02}/{timestamp}.npz"
        try: d = (np.load(path))
        except:
            d = (np.load(path.replace("huge", "fast")))
        #wh_pr = [levels_medium.index(x) for x in levels_ecm2]
        #d["pr"] = d["pr"][:720]
        d = {"sfc": d["sfc"]}
        d["sfc"] = d["sfc"][:720]
        extra = ["logtp", "15_msnswrf", "45_tcc", "168_2d"]
        ex = []
        for e in extra:
            a = np.load("/huge/proc/era5/extra/%s/%04d%02d/%d.npz" % (e, dt.year, dt.month, timestamp))["x"]
            if a.shape[0] == 1:
                a = a[0]
            ex.append(a[:,:,np.newaxis])
        d["sfc"] = np.concatenate([d["sfc"]] + ex, axis=-1)
        return d
    
    def index(self, data, col):
        return data[:, self.COLUMNS.index(col)]

    StationName = bytes # string
    RawData = np.array # (num_observations, num_measurements_per_obs)
    IsValidation = bool
    Output = Tuple[Any, RawData, List[StationName], List[IsValidation], int, np.array]
    def __getitem__(self, idx: int) -> Output:

        # Compute time bounds corresponding to this index.
        dt = self.datetimes[idx]
        return self.get_by_date(dt)

    def interpget(self, src, toy, hr):
        #return np.memmap("/huge/consts/"+'/%s/%d_%d.npy' % (src, toy, hr), dtype=np.float32, shape=(720,1440))# - a) / b)
        return np.load("/huge/consts/"+'/%s/%d_%d.npy' % (src, toy, hr), mmap_mode='r')

    def get_by_date(self, dt, do_era5=True):

        # Update the rounding file if necessary.
        #if dt.year != self.cur_year:
        #    self.cur_year = dt.year
        try: rounding = self.load_rounding_weights(dt.year)
        except: rounding = None
        
        # Load the data.
        prev = dt - timedelta(hours=1)
        data_file = f"{BASE}/{dt.year}/{dt.month:02}/{dt.year}{dt.month:02}{dt.day:02}{dt.hour:02}.npz"
        #print("uhh", data_file)
        data = np.load(data_file)["data"]
        #print("og", data.shape)
        """
        prev_file = f"{BASE}/{prev.year}/{prev.month:02}/{prev.year}{prev.month:02}{prev.day:02}{prev.hour:02}.npz"
        data_prev = np.load(prev_file)["data"]
        data_prev[:, 0] -= 3600
        print(data[:,0].min(), data[:,0].max())
        print(data_prev[:,0].min(), data_prev[:,0].max())
        """

        # combine data and data prev
        start = utctimestamp(dt - timedelta(seconds=self.dt))
        end = utctimestamp(dt + timedelta(seconds=self.dt))
        len_new = data.shape[0]

        if rounding is not None:
            data_file2 = f"/huge/proc/hres_consolidated/consolidated/{dt.year}/{dt.month:02}{dt.day:02}.npy"
            data2 = np.load(data_file2)
            ts2 = data2[:, 0]
            filt = (ts2 >= start) & (ts2 <= end)
            data2 = data2[filt, :]
            data2[:,0] -= (dt-datetime(1970,1,1)).total_seconds()
            data2[:, 1] += self.nnew # old stations go after the new ones
            data2[:, 2] /= 100. # hectopascals in the new format
            data2[:, 3] -= 273.15 # celsius in the new format
            data2[:, 4] -= 273.15 # celsius in the new format
            ucomp = data2[:, 5].copy()
            vcomp = data2[:, 6].copy()
            data2[:, 5] = vcomp
            data2[:, 6] = ucomp
            data2[:, 7] *= 0.3048 # feet to meters
            data2[:, 8] *= 1.60934 # miles to km
            data2 = np.hstack((data2, np.zeros((data2.shape[0], 1), dtype=np.float32)+np.nan))
            #print(data2[:,0].min(), data2[:,0].max())

            len_old = data2.shape[0]
            #print("now", data.shape, "old", data2.shape)
            data = np.concatenate((data, data2), axis=0)
        else:
            len_old = 0
        
        # Pull data. 
        timestamp = self.index(data, "timestamp")
        #filt = (timestamp >= start) & (timestamp <= end)
        filt = (timestamp >= -self.dt) & (timestamp <= self.dt)
        #print("prev", data.shape)
        relevant_data = data[filt, :]
        #print("Relevant", relevant_data.shape)
        #assert data.shape[0] == relevant_data.shape[0]
        out = np.empty((relevant_data.shape[0], len(self.variables)), dtype=np.float32)
        for i, f in enumerate(self.variables):
            out[:, i] = self.index(relevant_data, f)
            if f in ["tmpf", "dwpf"]:
                out[:, i] += 273.15
            if f == "mslp":
                out[:, i] *= 100
            out[:, i] = (out[:, i] - self.normalization[f][0]) / np.sqrt(self.normalization[f][1])
        out = torch.tensor(out)
        
        # Get station metadata.
        stations = self.index(relevant_data, "station").astype(np.int32)
        #print("Hey len stations", len(stations))

        # Get ERA5 data.
        if do_era5:
            era5_data = HresDataset.load_era5(dt)
        else:
            era5_data = None

        if rounding is not None:
            perc_rounded = rounding[stations]

            is_imprecise = perc_rounded > 0.5
            is_imprecise += perc_rounded < 0.025
        else:
            is_imprecise = np.zeros(len(stations), dtype=bool)

        def f(pt):
            lat, lon = pt
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return np.inf
            if round(lat, 2) == round(lat, 3) and round(lon, 2) == round(lon, 3):
                return 5
            else:
                return 0

        #print(threading.get_ident(), "got stations", indices[:10])
        is_fuzzyloc = np.array([f(self.stations_latlon_ext[idx]) for idx in stations])
        is_sl = np.array([5 if self.is_slop[idx] else 0 for idx in stations])

        # zero out slop winds. they're bad somehow
        out[is_sl != 0, 3] = np.nan
        out[is_sl != 0, 4] = np.nan
        assert out.shape[1] == 8

        weights = 1/(1 + is_fuzzyloc + is_sl + 5 * is_imprecise)

        #era5pr = era5_data["pr"]
        era5sfc = era5_data["sfc"]

        soy = dt.replace(month=1, day=1)
        toy = int((dt - soy).total_seconds()/86400)

        Rad = self.interpget("neoradiation_1", toy, dt.hour)
        Ang = self.interpget("solarangle_1", toy, dt.hour)#, a=0, b=180/np.pi)


        pts = []
        K = self.batch_size
        MARGIN = 3
        PAD = 0.5
        ret = []
        dics = []
        nn = 0
        while nn < K:
            # get a random station

            for attempt in range(10*K):
                st = np.random.choice(stations)
                #if i == 0: st = min([((self.stations_latlon_ext[st][0] - 37.44)**2+(self.stations_latlon_ext[st][1]+122.14)**2, st) for st in stations])[1]
                lat, lon = self.stations_latlon_ext[st]

                if lat >= 65 or lat <= -60: continue # i hate penguins
                if 0 <= lon <= MARGIN//2+1 or 360-(MARGIN//2+1) <= (lon+360) <= 360: continue # i hate edge cases. incidentally i hate spain
                #print("picked", lat, lon)
                # TODO: filter out stations that are too close to each other

                bad = False
                for pt in pts:
                    if abs(lat - pt[0]) + abs(lon - pt[1]) < 2 * MARGIN + 1:
                        bad = True
                if not bad: break
            #assert attempt < 48

            pts.append((lat, lon))
            baselat = int(np.round(lat + MARGIN / 2))
            baselon = int(np.round(lon - MARGIN / 2))
            baselonp = (baselon + 360) % 360

            dic = {}
            for arrname in ["Elevation30", "Modis30", "era5sfc", "Rad", "Ang", "Static_sfc"]: # era5pr
                arr = eval(arrname)
                ires = arr.shape[0]//180
                idxlat = (90-baselat) * ires
                idxlon = (baselonp) * ires
                #print("hey uh", arrname, arr.shape, ires, idxlat, idxlon)
                sub = arr[idxlat:idxlat+MARGIN*ires, idxlon:idxlon+MARGIN*ires].copy()
                if arrname.startswith("Modis"):
                    if baselat < -65:
                        sub = sub*0 + 15 # ?? idk jsut copied from elsewhere
                    sub[sub == -1] = 17
                    sub[sub == 0] = 17
                    sub -= 1
                    #print("uhhh sub", sub.min(), sub.max())
                    assert sub.min() >= 0
                    assert sub.max() < 17
                if arrname.startswith("Elevation"):
                    subs = [sub * 0.001, sub*0 + np.mean(sub)*0.001, sub*0 + np.std(sub)*0.005, (sub - np.mean(sub))/(1+np.std(sub))]
                    sub = np.stack(subs, axis=-1).astype(np.float16)
                if arrname == "Rad":
                    sub = ((sub-300)/400).astype(np.float16)[:,:,None]
                if arrname == "Ang":
                    sub = sub * np.pi / 180
                    sub = np.stack([np.cos(sub), np.sin(sub)], axis=-1).astype(np.float16)
                if arrname == "Static_sfc":
                    sub = torch.tensor(sub).half()
                dic[arrname] = torch.tensor(sub)
            #print("base", baselat, baselon) 
            wh = []
            whcoords = []
            for i, st in enumerate(stations):
                lat, lon = self.stations_latlon_ext[st]
                if abs(lat - (baselat - MARGIN/2)) <= MARGIN/2 - PAD and abs(lon - (baselon + MARGIN/2)) <= MARGIN/2 - PAD:
                    wh.append(i)
                    # ok this is cursed stuff
                    # we use F.grid_sample with align_corners=True
                    # means top left is (-1,-1). first index corresponds to longitude, second to latitude
                    full = dic["Elevation30"].shape[0] # this is the size of image. it corresponds to MARGIN degrees
                    res = MARGIN/full
                    # now (-1 + 2/full) will give you the second pixel, i.e. one resolution away
                    away = (lon - baselon)/res
                    coordlon = -1 + 2*away/(full-1)
                    # as a sanity check, consider longitude = baselon + MARGIN - res (end of the rectangle)
                    # that's the same as longitude = baselon + (full-1)*res = (full-1)*MARGIN/full = MARGIN-MARGIN/full = MARGIN-res
                    # now away = (MARGIN - res)/res. and so coordlon = -1 + 2*(MARGIN-res)/res/(full-1)
                    # -1 + 2*MARGIN/res/(full-1)  - res/res/(full-1) = -1 + 2*MARGIN/(MARGIN/full)/(full-1) - res/res/(full-1)
                    # -1 + 2* full/(full-1) - 1/(full-1) = -1 + 2 * (full - 1)/(full-1) = -1+2 = 1
                    # as expected, trivially. fml

                    # now let's do the same for latitude
                    away = (baselat - lat)/res
                    coordlat = -1 + 2*away/(full-1)
                    assert -1 <= coordlat <= 1
                    assert -1 <= coordlon <= 1
                    whcoords.append((coordlon, coordlat))
            outx = out[wh].half()
            stationsx = stations[wh]
            weightsx = weights[wh]
            if (~np.isnan(outx)).sum() < 5:
                pts = pts[:-1]
                continue
            
            weightsx = torch.tensor(weightsx.astype(np.float16))
            whcoords = torch.tensor(whcoords).float()
            ret.append((outx, whcoords, weightsx))
            dics.append(dic)
            nn += 1
            """
            nn = [((baselat-lat)*120, (lon-baselon)*120) for lat, lon in whcoords]
            import matplotlib.pyplot as plt
            plt.imshow(dic["Modis30"])
            plt.scatter([x[1] for x in nn], [x[0] for x in nn], s=0.5, color='red')
            plt.savefig("pp.png", bbox_inches='tight', dpi=300)
            # nn = [((38-lat)*120, (lon+123)*120) for lat, lon in whcoords]
            # plt.scatter([x[1] for x in nn], [x[0] for x in nn])
            import pdb
            pdb.set_trace()
            """
        assert len(ret) == K
        Dic = {}
        for k in dics[0]:
            Dic[k] = torch.cat([x[k][None] for x in dics], axis=0)   

        return ret, Dic


        # Select a random subset of batch_size stations for out, stations, and is_valid:
        #perc_rounded = rounding
        if self.batch_size is not None:
            pc_old = max(self.old_fraction, len_old / (len_old + len_new))
            pc_new = 1-pc_old
            weights = np.ones(len(self.stations_latlon_ext), dtype=np.float32)
            weights[self.is_slop] = 0.1
            weights = weights[stations]
            wnew = np.sum(weights[stations < self.nnew])
            nold2 = np.sum(stations >= self.nnew)
            weights[stations >= self.nnew] = pc_old * wnew / pc_new / nold2
            indices = np.random.choice(np.arange(len(stations)), size=self.batch_size, replace=False, p=weights/np.sum(weights))
            #print("target pc_old", pc_old, len(stations))
            out = out[indices]
            stations = stations[indices]
            #print("empirical pc old", np.sum(stations >= self.nnew) / len(stations), "empirical slop", np.sum(self.is_slop[stations]) / len(stations))
        
        if rounding is not None:
            perc_rounded = rounding[stations]

            is_imprecise = perc_rounded > 0.5
            is_imprecise += perc_rounded < 0.025
        else:
            is_imprecise = np.zeros(len(stations), dtype=bool)

        def f(pt):
            lat, lon = pt
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return np.inf
            if round(lat, 2) == round(lat, 3) and round(lon, 2) == round(lon, 3):
                return 5
            else:
                return 0

        #print(threading.get_ident(), "got stations", indices[:10])
        is_fuzzyloc = np.array([f(self.stations_latlon_ext[idx]) for idx in stations])
        is_sl = np.array([5 if self.is_slop[idx] else 0 for idx in stations])

        # zero out slop winds. they're bad somehow
        out[is_sl != 0, 3] = np.nan
        out[is_sl != 0, 4] = np.nan
        assert out.shape[1] == 8

        weights = 1/(1 + is_fuzzyloc + is_sl + 5 * is_imprecise)

        if any([x is None for x in [era5_data, out, stations, weights]]):
            with open("/tmp/aaaaa", "w") as f:
                f.write("gdi "+str([era5_data, out, stations, weights]))

        return era5_data, out, stations, weights, (dt - datetime(1970,1,1)).total_seconds()

if __name__ == '__main__':
    d = HresDataset(1000)
    #print(datetime(1970,1,1)+timedelta(seconds=int(d[-2001][-1])))
    from torch.utils.data import DataLoader
    dd = iter(DataLoader(d, batch_size=1, num_workers=1))
    a = next(dd)
    #import pdb; pdb.set_trace()
    (d.get_by_date(datetime(2019,6,21, 1)))
