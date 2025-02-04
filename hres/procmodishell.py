from osgeo import gdal, osr
import multiprocessing
from tqdm import tqdm
import time
import numpy as np
import os


def gps_dist(lat1, lon1, lat2, lon2):
    rads = np.radians
    lon1, lat1, lon2, lat2 = rads(lon1), rads(lat1), rads(lon2), rads(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

base = "/fast/ignored/modis/2020"


mn = 1e9
ls = os.listdir(base)
#ls = ["MCD12Q1.A2020001.h07v03.061.2022171161133.hdf"]
#ls = ["MCD12Q1.A2020001.h25v03.061.2022172020229.hdf"]
#ls = ["MCD12Q1.A2020001.h08v05.061.2022171160153.hdf"]
output = np.zeros((86400, 172800), dtype=np.int8) + 17
np.save("/dev/shm/arr.npy", output)

def proc(f):
    global extra
    print("extra", extra)
    moutput = np.load("/dev/shm/arr.npy", mmap_mode='r+')
    dataset = gdal.Open(base + "/" + f, gdal.GA_ReadOnly)

    subdataset = dataset.GetSubDatasets()[0][0] # L1 shit
    data = gdal.Open(subdataset, gdal.GA_ReadOnly)

    proj_info = data.GetProjection()
    source_srs = osr.SpatialReference()
    source_srs.ImportFromWkt(proj_info)

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(source_srs, target_srs)

    """
    inverse_transform = osr.CoordinateTransformation(target_srs, source_srs)

    # Get geotransform from the dataset
    geotransform = data.GetGeoTransform()
    inv_geotransform = gdal.InvGeoTransform(geotransform)
    """

    geotransform = data.GetGeoTransform()
    x_size = data.RasterXSize
    y_size = data.RasterYSize

    arr = data.ReadAsArray()
    assert (arr != 0).all()
    # Example: Transform the coordinate of the upper left corner of the raster
    #print("hey", x_size, y_size)
    xs = np.arange(x_size)
    ys = np.arange(y_size)
    X, Y = np.meshgrid(xs, ys)
    x_pixel, y_pixel = 0, 0
    x_coord = geotransform[0] + X * geotransform[1] + Y * geotransform[2]
    y_coord = geotransform[3] + X * geotransform[4] + Y * geotransform[5]

    """
    lon, lat, _ = transform.TransformPoint(x_coord[0,0], y_coord[0,0])
    d = gps_dist(lat, lon, 37.7544, -122.4387)

    lon, lat = -122.4387, 37.7544
    #lon, lat = lat, lon

    # Apply inverse transformation to get map coordinates
    map_x, map_y, _ = inverse_transform.TransformPoint(lat, lon)

    # Use inverse geotransform to convert map coordinates to pixel coordinates
    pixel_x, pixel_y = gdal.ApplyGeoTransform(inv_geotransform, map_x, map_y)
    pixel_x, pixel_y = int(pixel_x), int(pixel_y)  # Convert to integer pixel indices

    if 0 <= pixel_x < 2400 and 0 <= pixel_y < 2400:
        print(f, "pixel", pixel_x, pixel_y)
    """

    res = 0.002083333333333333333333

    for i in range(x_size):
            for j in range(y_size):
                lat, lon, _ = transform.TransformPoint(x_coord[i,j], y_coord[i,j])
                lonx = lon
                if lonx < 0: lonx += 360

                latidx = int(round((90 - lat)/res))
                lonidx = int(round(lonx/res))
                moutput[latidx-(extra-1):latidx+extra, lonidx-(extra-1):lonidx+extra] = arr[i, j]
                #for di in range(-extra, extra+1):
                #    for dj in range(-extra, extra+1):
                #            output[latidx+di, lonidx+dj] = arr[i, j]
                continue

                dat = arr[i, j]
                if i == pixel_y and j == pixel_x:
                    print("respective", lat, lon, dat)
                """
                d = gps_dist(lat, lon, 37.7544, -122.4387)
                if d < 0.1:
                    print("dist", d, i, j)
                """
    return

    print("huh", transform.TransformPoint(x_coord[pixel_y, pixel_x], y_coord[pixel_x, pixel_x]))

    print("data", arr[pixel_y, pixel_x])


    if d < mn:
        print(f, "dist", d)
        mn = d

    return


    print(x_coord, x_coord.shape)

    # Apply transformation
    t0 = time.time()
    for i in range(x_size):
        for j in range(y_size):
            lon, lat, _ = transform.TransformPoint(x_coord[i,j], y_coord[i,j])

    print(f"Longitude: {lon}, Latitude: {lat}", time.time()-t0)
    exit()

for extra in [16, 4, 1]:
  #for f in tqdm(ls):
  #pool.map(proc, ls)
  pool = multiprocessing.Pool(32)
  r = list(tqdm(pool.imap(proc, ls), total=len(ls)))
  pool.close()

"""
invalid = output == 0
from scipy import ndimage as nd
print("before")
print(invalid.sum())
print("transforming")
ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
output = output[tuple(ind)]
print("done")
print((output==0).sum())
"""
output = np.load("/dev/shm/arr.npy", mmap_mode='r+')
print("hey output", output)

np.save("/fast/ignored/modis/2020.npy", output)
print("saved!")


top = 38.1154
left = -123.201 + 360
bot = 37.096
right = -121.44 + 360

topidx = int(round((90 - top)/res))
botidx = int(round((90 - bot)/res))
leftidx = int(round(left/res))
rightidx = int(round(right/res))

bay = output[topidx:botidx, leftidx:rightidx]
print("hey bay", bay)
import matplotlib.pyplot as plt
plt.imshow(bay)
plt.savefig('bay2.png', bbox_inches='tight')
