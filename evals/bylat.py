import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

levels_joank = [10, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 925, 950, 975, 1000]
levels_medium = [10, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000]

def mk(n, h, v, l=None):
    base = '/huge/deep/evaluation/%s/errors/'%n
    ls = [x for x in os.listdir(base) if '.bylat.' in x and "+%d."%h in x]
    alldata = []
    for f in tqdm(ls):
        with open(base+f, 'rb') as file:
            data = pickle.load(file)
            assert data["rmse"][v].shape[1] == {'graphbs_yolo': len(levels_joank), 'joansucks': len(levels_medium)}[n]
            alldata.append(data["rmse"][v][:, l])
    alldata = np.array(alldata)
    rms = np.sqrt(np.mean(np.square(alldata), axis=0))/np.mean(alldata)
    lats = np.arange(90, -89.99, -0.25)
    plt.plot(lats, rms, label=n)

pp = [("129_z", 24), ("129_z", 48), ("130_t", 24), ("130_t", 48)]
for vn, dt in pp:
    plt.clf()
    plt.cla()
    mk("graphbs_yolo", dt, vn, levels_joank.index(500))
    mk("joansucks", dt, vn, levels_medium.index(500))
    plt.legend()
    plt.xlabel('latitude')
    plt.ylabel('RMSE/mean error')
    plt.title('%s %dh' % (vn, dt))
    plt.tight_layout()
    plt.savefig('bylat_%s_%d.png'%(vn, dt), dpi=300, bbox_inches='tight')