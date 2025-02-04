from utils import *
from meshes import *
from data import *

meshes = []
for source in ['gfs-28','era5-28','hres-13']:
    mesh = LatLonGrid(source=source)
    meshes.append(mesh)

dataconf = NeoDataConfig(inputs=[meshes[0]],
                         ouputs=meshes[1:],
                         requested_dates = get_dates((D(1900, 1, 1),D(2100, 1, 1)))
                         )
data = NeoWeatherDataset(dataconf)
data.check_for_dates()