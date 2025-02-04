import numpy as np
import scipy.stats

print("loading")
f = np.load("/fast/ignored/modis/2020.npy")
print("done")

f.shape = (21600, 4, 43200, 4)

f = np.transpose(f, (0, 2, 1, 3))
print("trans")
"""
mid = f[:,:,1,1]
np.save("/fast/ignored/modis/2020_small3.npy", mid)
print("saved")
"""
f = f.reshape((21600, 43200, 16))
print("resh")



#smol = np.median(f, axis=(1,3)).astype(np.int8)
smol = scipy.stats.mode(f, axis=2).mode.astype(np.int8)
print("done!")
np.save("/fast/ignored/modis/2020_small2.npy", smol)
print("saved")