#%%
import json
from dateutil import parser  
import pytz
from matplotlib import pyplot as plt
plt.style.use('/fast/haoxing/deep/wb.mplstyle')
import numpy as np
from scipy.ndimage import gaussian_filter1d as gf
import pickle
import os
import pandas as pd
from types import SimpleNamespace
from datetime import datetime, timezone
import sys
sys.path.append('/fast/wbhaoxing/deep')
from evals import *
from utils import *
vardict = {
    '129_z_500': ['z_500','500mb Geopotential','m^2 s^-2'],
    '129_z_850': ['z_850','850mb Geopotential','m^2 s^-2'],
    '133_q_500': ['q_500','500mb Specific Humidity','ug/kg'],
    '133_q_850': ['q_850','500mb Specific Humidity','ug/kg'],
    '131_u_500': ['u_500','500mb U Wind','m s^-1'],
    '132_v_500': ['v_500','500mb V Wind','m s^-1'],
    '165_10u': ['10u','10m U Wind','m/s'],
    '166_10v': ['10v','10m V Wind','m/s'],
    '167_2t': ['2t','2m Temperature','K'],
    '151_msl': ['msl','Mean Sea Level Pressure','Pa'],
}

EVALUATION_PATH = '/huge/deep/evaluation/'

def gfq(x,n):
    return np.sqrt(gf(np.square(x),n))

#with open('/fast/wbjoan5/deep/bydate.pickle','rb') as f:
with open('/fast/wbjoan5/deep/bydate_Nov6.pickle','rb') as f:
    jfs = pickle.load(f)




def plot_val(t,y,name,lw=2,ci=0):
    plt.plot(t,y,c='C%d'%ci,alpha=0.2,linewidth=lw)
    plt.plot(t,gfq(y,14),c='C%d'%ci,label=name,linewidth=lw)

allvars = next(iter(jfs.values())).keys()
#pirint(allvars)


def get_weaterbench_scores(d):
    if not ('GraphCast' in d['name'] or 
        'Pangu' in d['name'] or 
        'IFS HRES vs Analysis' in d['name']):
        return None

    print(d['name'])
    t = [parser.parse(t) for t in d['x']]
    y = np.array(d['y'],dtype=np.float32)
    idx = np.where(~np.isnan(y))[0]
    t = [t[i] for i in idx]
    y = y[idx]
    return t,y

def get_weatherbench_data(var):
    # Q: how do i add more variables from weatherbench?
    # A: https://chat.windbornesystems.com/#narrow/stream/201-tech-dlnwp/topic/compound.20timestep/near/2866302
    wb_path = 'misc/wb_jsons/%s.json'%fmt[0]
    if not os.path.exists(wb_path):
        return None
    with open(wb_path,'r') as f:
        raw = json.load(f)
    wb_data = raw['response']['graph']['figure']['data']
    return wb_data

cache = {}

def get_model_scores(model,var,dt=24,vs='era5-28',metric='rmse'):
    global cache
    dates = []
    vars = []
    for fn in sorted(os.listdir(get_errors_path(model))):
        fns = fn.split('.')
        if len(fns) < 3: continue
        #print("heyyy", fns)
        if not fns[0].endswith(f'+{dt}') or fns[1] != 'vs_'+vs or fns[2] != 'json':
            #print("continuing", fn, dt, vs)
            continue
        #with open(get_errors_path(model)+fn,'r') as f:
        #    j = json.load(f)
        pp = get_errors_path(model)+fn
        if pp in cache:
            j = cache[pp]
            #print("from cache!", pp)
        else:
            with open(pp,'r') as f:
                j = json.load(f)
            cache[pp] = j
        rms = j[metric]
        #print(rms)
        date = datetime.strptime(fn.split('+')[0],'%Y%m%d%H').replace(tzinfo=timezone.utc)
        if (date >= datetime(2024,3,10,tzinfo=pytz.utc) and date <= datetime(2024,5,31,tzinfo=pytz.utc)) or "neoTar" in model or "22fct" in model:
            dates.append(date)
            vars.append(rms[var])
    # sort both dates and vars by dates
    #print(dates,vars)
    try:
        dates, vars = zip(*sorted(zip(dates, vars)))
    except:
        print(f"Could not get scores for model={model} var={var} dt={dt} vs={vs}")
    return dates, vars

#cdict = {os.path.basename(f).split('_')[0]:xr.open_dataset(f) for f in cfs}
cdict = {
    "hres_2024": json.load(open("/huge/users/haoxing/ifs/hres_vs_era5_2024marmay.json", "r")),
    "gfs_2024": json.load(open("/huge/users/haoxing/gfs/gfs_vs_era5_2024marmay.json", "r")),
    "aurora_v_era5_2024": json.load(open("/huge/users/haoxing/aurora/aurora_vs_era5_2024marmay4.json", "r")),
    #"aurora_v_hrest0_2024": json.load(open("/huge/users/haoxing/aurora/aurora_vs_hrest0_2024marmay.json", "r")),
}

NDAY = 14
NORM = 0

display_names = {
    "hres_2024": "IFS HRES",
    "gfs_2024": "NOAA GFS",
    "aurora_v_era5_2024": "Microsoft Aurora",
    "aurora_v_hrest0_2024": "Aurora (vs HRES T0)",
    #'hegelquad_333M': "Old WeatherMesh (Aug-Dec 2022)",
    'rtyamahabachelor5_328M': "WeatherMesh-2",
}

for metric in ['rmse']:#,'bias','mae']:
    print(metric)
    vars = ['167_2t','165_10u','166_10v','151_msl','129_z_500','129_z_850','133_q_500','131_u_500','132_v_500']
    fig, axs = plt.subplots(3, 3,figsize=(15,12),dpi=200)
    # fig.patch.set_alpha(0)
    # for ax in axs.flatten():
    #     ax.patch.set_alpha(0)
    dts = np.arange(1,int(NDAY*24/6+1))*6
    dts = np.arange(1,int(NDAY*24/12+1))*12
    # dts = [24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216]
    # dts = list(range(0, 25))
    dts = list(range(24, 337, 24))
    print(dts)
    for i,v in enumerate(vars):
        ax = axs.flatten()[i]
        try: level = int(v.split('_')[-1]) ; vs = '_'.join(v.split('_')[:-1])
        except: level = None; vs = v

        colors = {
            "hres_2024": "tab:blue",
            "gfs_2024": "tab:orange",
            "aurora_v_era5_2024": "tab:purple",
            "aurora_v_hrest0_2024": "tab:purple",
        }

        for cmodel, errors in cdict.items():
            rms2 = errors[v]
            n_days = len(rms2)
            dts2 = list(range(24, 24*n_days+1, 24))
            sel = {'region':'global','metric':metric}
            if level is not None: sel['level'] = level
            if cmodel == "hres":
                norm = rms2, dts2
            #print("hey norm is", norm, "rms2", rms2.shape, norm[0].shape)
            if NORM:
                ax.plot(dts2,(rms2-norm[0])/norm[0]*100,label=display_names[cmodel])
            else:
                if cmodel == "aurora_v_hrest0_2024":
                    ax.plot(dts2,rms2,label=display_names[cmodel], linestyle='--')
                else:
                    ax.plot(dts2,rms2,label=display_names[cmodel])
            ax.set_title(vardict[v][1])
            # label x axis with days
            ax.set_xticks([x*24 for x in [1,3,7,10,14]])
            ax.set_xticklabels([1,3,7,10,14])
            ax.set_xlabel('Days')
            ax.set_ylabel(f'{metric.upper()} ({vardict[v][2]})')
            #ax.grid(1)
        
        for modelname in ['rtyamahabachelor5_328M']:
            def get_avg_rms(dt):
                s = get_model_scores(modelname,v,dt=dt,metric=metric)
                rms = np.array(sorted(s[1]))
                rms = rms[3:-3]
                stderr = np.std(rms)/np.sqrt(len(rms))
                return np.mean(np.square(rms))**0.5, stderr
            rms = np.array([get_avg_rms(dt)[0] for dt in dts])
            stderr = np.array([get_avg_rms(dt)[1] for dt in dts])
            if '133_q' in v: 
                rms /= 1e6
                stderr /= 1e6
            if NORM:
                idx = [list(norm[1]).index(a) for a in dts]
                ax.plot(dts,(rms-norm[0][idx])/norm[0][idx]*100,label=display_names[modelname])
            else: 
                ax.plot(dts,rms,label=display_names[modelname])#, color='tab:green')
                # ax.errorbar(dts, rms, yerr=stderr, label=display_names[modelname])
        # ax.legend()
        # legend = ax.legend()
        # legend.get_frame().set_facecolor('#E3E8E4')  # Transparent legend background
        # legend.get_frame().set_alpha(1)

    # set overall title
    #fig.suptitle(f'WeatherMesh 2.0 vs Operational Models (March-May 2024)\nProbably aurora here is wrong, but maybe it is right??', fontsize=16)
    fig.suptitle(f'WeatherMesh-2 vs Operational Models (March-May 2024)', fontweight='bold', fontsize=22)
    fig.tight_layout()
    plt.savefig(f'ignored/metrics_aurora.png')
