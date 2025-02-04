from osgeo import gdal
import numpy as np

def pixel_to_coord(x, y, geotransform):
    """Convert from raster coordinate space to geographic coordinate space."""
    x_geo = geotransform[0] + geotransform[1] * x + geotransform[2] * y
    y_geo = geotransform[3] + geotransform[4] * x + geotransform[5] * y
    return (x_geo, y_geo)

dataset = gdal.Open("/fast/ignored/elevation/mn75_grd/z001003.adf")
elevation = np.zeros((86400, 172800), dtype=np.float32)
geotransform = dataset.GetGeoTransform()
print(geotransform)
print("Huh", pixel_to_coord(46086, 41725, geotransform))
array = dataset.ReadAsArray()
print("huh", array)
print(array.shape, array.max())
print(np.argmax(array), np.unravel_index(np.argmax(array), array.shape))
elevation[2880:2880+array.shape[0], :] = np.roll(array, array.shape[1]//2, axis=1)
print(elevation, elevation.min(), elevation.max())
np.save("/fast/ignored/elevation/mn75.npy", elevation)

exit()

def pixel_to_coord(x, y, geotransform):
    """Convert from raster coordinate space to geographic coordinate space."""
    x_geo = geotransform[0] + geotransform[1] * x + geotransform[2] * y
    y_geo = geotransform[3] + geotransform[4] * x + geotransform[5] * y
    return (x_geo, y_geo)

for f in ["mn30", "sd30"]:
    dataset = gdal.Open("/fast/ignored/elevation/%s_grd/w001001.adf"%f)
    elevation = np.zeros((21600, 43200), dtype=np.float32)
    array = dataset.ReadAsArray()
    elevation[-array.shape[0]:, :] = np.roll(array, array.shape[1]//2, axis=1)
    print(f, elevation, elevation.min(), elevation.max())
    np.save("/fast/ignored/elevation/%s.npy"%f, elevation)