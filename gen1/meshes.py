import base64
import hashlib
import icosphere
import json
import numpy as np
import os
import pickle
from scipy.spatial import KDTree

from gen1.utils import (
    SourceCodeLogger, 
    NeoDatasetConfig, 
    levels_full, 
    load_state_norm, 
    CONSTS_PATH,
)

def haversine(lon1, lat1, lon2, lat2):
    from math import sin, cos, sqrt, atan2, radians
    R = 6371  # Radius of Earth in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2) * sin(dlat/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2) * sin(dlon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

class LatLonGrid(NeoDatasetConfig,SourceCodeLogger):
    def __init__(self,**kwargs):
        super().__init__(self,**kwargs)

        self.type = "latlon"
        lats = np.arange(90, -90.01, -0.25)[:-1]
        lats.shape = (lats.shape[0]//self.subsamp, self.subsamp)
        self.lats = np.mean(lats, axis=1)

        lons = np.arange(0, 359.99, 0.25)
        lons.shape = (lons.shape[0]//self.subsamp, self.subsamp)
        lons[lons >= 180] -= 360
        self.lons = np.mean(lons, axis=1)
        self.parent = None
        self.bbox = None
        self.update_mesh()

    def update_mesh(self):
        self.Lons, self.Lats = np.meshgrid(self.lons, self.lats)
        self.res = 0.25 * self.subsamp

        self.weights = np.cos(self.Lats * np.pi/180)

        self.Lons /= 180
        self.Lats /= 90
        self.xpos = np.stack((self.Lats, self.Lons), axis=2)

        self.state_norm,self.state_norm_stds,self.state_norm_means = load_state_norm(self.wh_lev,self,with_means=True)

    
    def lon2i(self, lons):
        return np.argmin(np.abs(self.lons[:,np.newaxis] - lons),axis=0)
    
    def lat2i(self, lats):
        return np.argmin(np.abs(self.lats[:,np.newaxis] - lats),axis=0)
    
    def to_json(self):
        out = {}
        out['pressure_vars'] = self.pressure_vars
        out['sfc_vars'] = self.sfc_vars
        out['full_varlist'] = self.full_varlist
        out['levels'] = self.levels
        out['lats'] = self.lats.tolist()
        out['lons'] = self.lons.tolist()
        out['res'] = self.res
        st = json.dumps(out, indent=2)
        hash_24bit = hashlib.sha256(st.encode()).digest()[:3]
        base64_encoded = base64.b64encode(hash_24bit).decode()
        return st, base64_encoded

    @staticmethod
    def from_json(js):
        out = LatLonGrid()
        out.__dict__ = js
        out.source = "unknown"
        out.lats = np.array(out.lats)
        out.lons = np.array(out.lons)
        out.wh_lev = [levels_full.index(x) for x in out.levels] #which_levels
        out.subsamp = 1; assert np.diff(out.lats)[1] == -0.25, f'lat diff is {np.diff(out.lats)}'
        out.update_mesh()
        return out

class Mesh(SourceCodeLogger):
    def check_dists(self):
        pts = self.neighbors[0,:]
        #print(pts)
        #exit()
        lons = self.lons[pts]; lats = self.lats[pts]
        lon0, lat0 = lons[0], lats[0]
        distances_km = [haversine(lon0, lat0, lon, lat) for lon, lat in zip(lons[1:], lats[1:])]
        #print(distances_km)
        print("Min Neighbors Distance: %.1f km" % np.min(distances_km))
        print("Max Neighbors Distance: %.1f km" % np.max(distances_km))
        #exit()

    def __init__(self, n, k, levels,remove_poles=False):
        #print(n,k,levels)
        self.type = "icos"
        self.n = n
        self.k = k
        self.levels = None
        self.remove_poles = remove_poles

        self.vertices, self.faces = icosphere.icosphere(n)
        self.lats = np.arccos(self.vertices[:,2]) * 180 / np.pi - 90
        if remove_poles:
            good_lats = np.where(np.logical_and(-70 < self.lats, self.lats < 70))[0]
            self.vertices = self.vertices[good_lats]
            self.lats = self.lats[good_lats]
        self.lons = np.arctan2(self.vertices[:,0], self.vertices[:, 1]) * 180 / np.pi
        self.N = len(self.vertices)
        self.lons[self.lons < 0] += 360
        self.n_pr = len(levels) * 6
        self.n_levels = len(levels)
        self.n_pr_vars = 6
        self.n_sfc = 5
        self.n_sfc_vars = 5

        kdtree = KDTree(self.vertices)

        self.neighbors = []

        for i in range(self.N):
            dist, pts = kdtree.query(self.vertices[i], self.k)
            self.neighbors.append(pts.astype(np.int32))

        self.neighbors = np.array(self.neighbors)
        self.check_dists()

        pts = self.vertices[self.neighbors]
        delta = pts-self.vertices[:, np.newaxis, :]
        deltanorm = np.linalg.norm(delta, axis=2)[:,:,np.newaxis]
        ptslat = self.lats[self.neighbors, np.newaxis]/90
        ptslon = self.lons[self.neighbors, np.newaxis]/360
        deltalat = ptslat - self.lats[:, np.newaxis, np.newaxis]/90
        deltalon = ptslon - self.lons[:, np.newaxis, np.newaxis]/360
        self.xpos = np.concatenate([pts, delta, deltanorm, ptslat, ptslon, deltalat, deltalon], axis=-1)

    def normalize_xpos(self):
        print("std", np.std(self.xpos, axis=(0,1)))
        self.xpos /= np.array([0.5773503 , 0.57735024, 0.57735027, 0.02121295, 0.02121297, 0.02121301, 0.01313381, 0.43476311, 0.28929714,0.01265145, 0.05898692], dtype=np.float32)
        print("std", np.std(self.xpos, axis=(0,1)))

 

def get_mesh(n, k, levels=None):
    if levels is None:
        levels = levels_full


    fn = f"{CONSTS_PATH}/grids/grid_%d_%d_%d.pickle" % (n, k, len(levels))
    if os.path.exists(fn) and 0:
        with open(fn, "rb") as f:
            return pickle.load(f)

    g = Mesh(n, k, levels)

    g.wh_lev = [levels_full.index(x) for x in levels]

    os.makedirs(f"{CONSTS_PATH}/grids", exist_ok=True)
    #with open(fn, "wb") as f:
    #    pickle.dump(g, f)
    return g
