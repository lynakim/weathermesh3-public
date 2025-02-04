
# %% 
import tropycal.tracks as tracks
import matplotlib.pyplot as plt
import numpy as np
from evals.tc.tclib import * 

# %%
season = tracks.TrackDataset(basin='all',source='ibtracs').get_season(year=2022,basin='all')
season.plot()
plt.savefig('ignored/ohp.png')

# %% 
plt.clf()
for k,v in season.dict.items():
    if np.max(v['vmax']) > 100:
        print(v['name'],np.max(v['vmax']))
        storm = season.get_storm(k)
        storm.plot()
plt.savefig(f'ignored/ohp.png')

