import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import numpy as np
from shapely.geometry import Polygon

if 0:
    from model_latlon_3d import *
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(ForecastStepConfig([imesh], outputs=[omesh], Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,6,6), 
                                                    hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6,24], dims_per_head=32, 
                                                    processor_dt=6, use_matepoint=True, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,5)))
res = 0.25
lat_p = 8
lon_p = 8
lat_w = 5
lon_w = 5

nlat = int(180 / res)
nlon = int(360 / res)


plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

for i in range(nlat//lat_p):
    lat = i*lat_p*res - 90
    print(lat)
    ax.plot([-180, 180], [lat,lat], 'gray')

for i in range(nlon//lon_p):
    lon = i*lon_p*res - 180
    ax.plot([lon, lon], [-90,90], 'gray')


def box_at(x,y,size=7):
    return [x, x+size, x+size, x, x], [y+size, y+size, y, y, y+size]



def patch_tranform(lon_i,lat_i):
    lon_i = np.array(lon_i) ; lat_i = np.array(lat_i)
    return lon_i*lon_p*res - 180, lat_i*lat_p*res - 90



x,y = 40,60

box = patch_tranform(*box_at(x,y))
ax.plot(box[0], box[1], 'r',linewidth=2)
rect = Polygon(list(zip(*patch_tranform(*box_at(x+3,y+3,size=1)))))

ax.add_geometries([rect], ccrs.PlateCarree(), facecolor='red', alpha=0.5)

plt.tight_layout()
plt.savefig('ignored/test.png', dpi=300)


