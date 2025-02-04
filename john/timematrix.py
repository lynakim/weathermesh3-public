import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import time

path = '/huge/deep/runs/run_Jul29-bachelor_20240801-205915/dataloader/dates/'

run_name = path.split('/')[-4]
print(run_name)
procs = sorted(os.listdir(path))

print(procs)
num_procs = len(procs)
all_steps = []
run_len = None
arr = None
for proc in range(num_procs):
    steps = []
    with open(os.path.join(path,procs[proc]), 'r') as f:
        lines = f.readlines()
        if run_len is None:
            run_len = len(lines)
        #run_len = 1000
        for line in lines[:run_len]:
            step = line.split('step ')[1].split(': ')[0]
            dates = line.split(': ')[1].split(' ')
            if arr is None:
                arr = np.zeros((run_len,num_procs,1+len(dates)),dtype=np.int32) 
            timestamps = np.array([int(time.mktime(time.strptime(date.strip(), '%Y%m%d%H'))) for date in dates])
            arr[int(step),proc,:] = np.concatenate(([int(step)],timestamps))

arr = arr[:,:,1:]
print(arr.shape)

uniques = np.sort(np.unique(arr))[1:]
ind_arr = np.searchsorted(uniques, arr) - 1
D = int(np.ceil(np.sqrt(len(uniques))))
hits = np.zeros(D*D)


from matplotlib import cm
plt.switch_backend('Agg')  # Use Agg backend for faster rendering
cmap = cm.viridis  #
save_dir = '/fast/to_srv/dataload/'

for i in range(2000):
    print(i)
    hits[ind_arr[i,:].flatten()] = 1

    img = hits.reshape((D,D))
    plt.imsave(f'{save_dir}/step={i:05},_neovis.png', img, cmap=cmap)
    hits = hits * 0.95

import neovis
gif = neovis.Neovis(save_dir)
gif.make(anti_alias=False)


exit()

import imageio
# Collect all image file names
filenames = [f'tools/data/image_{i}.png' for i in range(run_len)]

# Create a GIF
with imageio.get_writer('output.gif', mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

