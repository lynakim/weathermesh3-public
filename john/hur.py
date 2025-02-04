import numpy as np
import matplotlib.pyplot as plt

path = '/fast/proc/era5/extra/tc-minp/200808/1218283200.npz'

data = np.load(path)['x']
plt.imsave('imgs/jtc.png',np.nan_to_num(data,nan=-10))

print(np.nanmin(data),np.nanmax(data),np.nanmean(data))
pass
