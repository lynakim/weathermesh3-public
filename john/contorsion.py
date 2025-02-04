import numpy as np
import matplotlib.pyplot as plt


path = '/fast/proc/era5/f000/199709/873072000.npz'
data = np.load(path)
print(data['pr'].shape)
pass