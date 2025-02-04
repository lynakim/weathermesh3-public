import json
from dateutil import parser  
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d as gf
import pickle
import os
import pandas as pd

vardict = {
    '129_z_500': ['z_500','500mb Geopotential Height','m^2 s^-2'],
    '129_z_850': ['z_850','850mb Geopotential Height','m^2 s^-2'],
    '133_q_500': ['q_500','500mb Specific humidity','ug/kg'],
    '133_q_850': ['q_850','500mb Specific humidity','ug/kg'],
    '165_10u': ['10u','10m U Wind','m s^-1'],
    '166_10v': ['10v','10m V Wind','m s^-1'],
    '167_2t': ['2t','2m Temperature','K'],
    '151_msl': ['msl','Mean Sea Level Pressure','Pa'],
}

def gfq(x,n):
    return np.sqrt(gf(np.square(x),n))

#with open('/fast/wbjoan5/deep/bydate_Nov6.pickle','rb') as f:
with open('/fast/wbjoan5/deep/bydate.pickle','rb') as f:
    jfs = pickle.load(f)




def plot_val(t,y,name,lw=2,ci=0):
    plt.plot(t,y,c='C%d'%ci,alpha=0.2,linewidth=lw)
    plt.plot(t,gfq(y,14),c='C%d'%ci,label=name,linewidth=lw)

allvars = next(iter(jfs.values())).keys()
#pirint(allvars)

allmeans = {}
for var in allvars:
    if not var in vardict:
        continue
    fmt = vardict[var]
    # Q: how do i add more variables from weatherbench?
    # A: https://chat.windbornesystems.com/#narrow/stream/201-tech-dlnwp/topic/compound.20timestep/near/2866302
    wb_path = 'misc/wb_jsons/%s.json'%fmt[0]
    if not os.path.exists(wb_path):
        continue
    with open(wb_path,'r') as f:
        raw = json.load(f)
    data = raw['response']['graph']['figure']['data']

    plt.figure(figsize=(12,6))
    plt.title(fmt[1])
    plt.ylabel(f'RMSE {fmt[2]}')



    j = 1
    means = {}
    for i,d in enumerate(data):
        if not ('GraphCast' in d['name'] or 
            'Pangu' in d['name'] or 
            'IFS HRES vs Analysis' in d['name']):
            continue

        print(d['name'])
        t = [parser.parse(t) for t in d['x']]
        y = np.array(d['y'],dtype=np.float32)
        idx = np.where(~np.isnan(y))[0]
        t = [t[i] for i in idx]
        y = y[idx]
        means[d['name']] = np.sqrt(np.mean(np.square(y)))
        if var.startswith('133_q'): means[d['name']] *= 1e6
        plot_val(t,y,d['name'],ci=j)
        j += 1


    t, all = zip(*sorted(jfs.items()))
    y = [x[var] for x in all]
    ci = 0
    plot_val(t,y,'WindBorne Nov6 vs ERA5',lw=3,ci=0)
    means['WindBorne vs ERA5'] = np.sqrt(np.mean(np.square(y)))
    allmeans[fmt[0]] = means


    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'ignored/{fmt[0]}.png')

df = pd.DataFrame(allmeans)   
print(allmeans)
print(df)
#df = df.map(lambda x: f'{x:.{4}g}' if isinstance(x, (int, float)) else x)

md = df.to_markdown()
with open('ignored/table_means.md','w') as f:
    f.write(md)
os.system('pandoc -s ignored/table_means.md -c table.css -o ignored/table_means.html')

print(df)
