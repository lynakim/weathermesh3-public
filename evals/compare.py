import json
from dateutil import parser  
import pytz
from matplotlib import pyplot as plt
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
    '129_z_500': ['z_500','500mb Geopotential Height','m^2 s^-2'],
    '129_z_850': ['z_850','850mb Geopotential Height','m^2 s^-2'],
    '133_q_500': ['q_500','500mb Specific humidity','ug/kg'],
    '133_q_850': ['q_850','500mb Specific humidity','ug/kg'],
    '165_10u': ['10u','10m U Wind','m s^-1'],
    '166_10v': ['10v','10m V Wind','m s^-1'],
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
        if (date >= datetime(2020,1,1,tzinfo=pytz.utc) and date <= datetime(2022,12,31,tzinfo=pytz.utc)) or "rtyblong" in model:
            dates.append(date)
            vars.append(rms[var])
    # sort both dates and vars by dates
    #print(dates,vars)
    if len(dates) == 0:
        return None, None
    #assert len(dates) > 0, f"No dates found for {model} {var} {dt}"
    dates, vars = zip(*sorted(zip(dates, vars)))
    return dates, vars

def get_model_scores_pickle(model,var,dt=24):
    dates = []
    vars = []
    for fn in sorted(os.listdir(get_errors_path(model))):
        if not fn.split('.')[0].endswith(f'+{dt}') or ('gfs' in fn) or ('era5' in fn):
            continue
        with open(get_errors_path(model)+fn,'rb') as f:
            rms = pickle.load(f)
        date = datetime.strptime(fn.split('+')[0],'%Y%m%d%H').replace(tzinfo=timezone.utc)
        if (date >= datetime(2020,1,1,tzinfo=pytz.utc) and date <= datetime(2020,12,31,tzinfo=pytz.utc)) or "neoTar" in model or "22fct" in model:
            dates.append(date)
            vars.append(rms[var])
    # sort both dates and vars by dates
    assert len(dates) > 0, f"No dates found for {model} {var} {dt}"
    dates, vars = zip(*sorted(zip(dates, vars)))
    return dates, vars

def compater_table():
    allmeans = {}
    for var in allvars:
        if not var in vardict:
            continue
        fmt = vardict[var]

        wb_data = get_weatherbench_data(var)
        
        scores = []

        for i,d in enumerate(wb_data):
            score = SimpleNamespace()
            ret = get_weaterbench_scores(d) 
            if ret is None: continue
            score.time, score.vals = ret
            if var.startswith('133_q'): score.vals *= 1e6
            score.name = d['name']
            scores.append(score)

        models = ['Pantene_287M','Neoreal_335M', 'Tardis_289M', 'Quadripede_218M', 'Dec11_TardisL2FT_289M', 'TardisNeoL2FT_289M', 'EnsSmall_166M', 'SmolMenace_311M', 'SingleDec_240M', 'NeoEnc_240M']
        models.extend(['Tiny_311M', 'TinyOper_311M', 'TinyOper22fct_311M', 'Tiny2_311M', 'TinyOper2_311M', 'TinyOper2_22fct_311M'])
        models.extend(['neoTardisNeoL2FT_289M', 'neoTardis22anl_289M', 'neoTardis22fct_289M', 'neoTardis22fct_luigi2_289M'])

        models = ['Tardis_289M', 'Quadripede_218M', 'neoquadripede_331M', 'Dec11_TardisL2FT_289M', 'TardisNeoL2FT_289M']
        models.extend(['Tiny_311M', 'TinyOper_311M', 'TinyOper22fct_311M', 'Tiny2_311M', 'TinyOper2_311M', 'TinyOper2_22fct_311M'])
        models.extend(['neoTardisNeoL2FT_289M', 'neoTardis22anl_289M', 'neoTardis22fct_289M', 'neoTardis22fct_luigi2_289M'])

        for i,model in enumerate(models):
            score = SimpleNamespace()
            score.time, score.vals = get_model_scores(model,var)
            score.name = model
            scores.append(score)

        score = SimpleNamespace()
        score.name = "WB_Nov6"
        t, all = zip(*sorted(jfs.items()))
        y = [x[var] for x in all]
        score.time = t
        score.vals = y
        scores.append(score)


        plt.figure(figsize=(12,6))
        plt.title(fmt[1])
        plt.ylabel(f'RMSE {fmt[2]}')

        means = {}
        for i,score in enumerate(scores):
            means[score.name] = np.sqrt(np.mean(np.square(score.vals)))
            plot_val(score.time,score.vals,score.name,ci=i)

        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{EVALUATION_PATH}/{fmt[0]}.png')
        allmeans[fmt[0]] = means

    df = pd.DataFrame(allmeans)   
    #print(allmeans)
    print(df)

import xarray as xr

wbpath = '/fast/weatherbench2_results/deterministic/'
cfs = [
        f'{wbpath}/hres_vs_analysis_2020_deterministic.nc', 
        #f'{wbpath}/ens_vs_analysis_2020_deterministic.nc', 
        #f'{wbpath}/climatology_vs_era_2020_deterministic.nc',
        f'{wbpath}/graphcast_vs_era_2020_deterministic.nc', 
        f'{wbpath}/pangu_vs_era_2020_deterministic.nc',
       ]

#cdict = {os.path.basename(f).split('_')[0]:xr.open_dataset(f) for f in cfs}
cdict = {os.path.basename(f).split('_vs_')[0]:xr.open_dataset(f) for f in cfs}
#with Plotter(open_browser=False,output_dir=f'/fast/to_srv/{modelname}/') as plt:

NDAY = 15
INTERVAL = 24
NORM = 1
#NORM = 0
for metric in ['rmse']:#,'bias','mae']:
    print(metric)
    vars = ['129_z_500','129_z_850','133_q_500','131_u_500','132_v_500','165_10u','166_10v','167_2t','151_msl']
    fig, axs = plt.subplots(3, 3,figsize=(15,12))
    dts = np.arange(1,int(NDAY*24/6+1))*6
    dts = np.arange(1,int(NDAY*24/12+1))*12
    dts = np.arange(1,int(NDAY*24/INTERVAL+1))*INTERVAL
    dts = [24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216]
    print(dts)
    for i,v in enumerate(vars):
        ax = axs.flatten()[i]
        try: level = int(v.split('_')[-1]) ; vs = '_'.join(v.split('_')[:-1])
        except: level = None; vs = v
        cv = ncar2cloud_names[vs]
        for cmodel,xar in cdict.items(): 
            sel = {'region':'global','metric':metric}
            if level is not None: sel['level'] = level
            rms2 = xar[cv].sel(**sel).values
            dts2 = xar[cv].sel(**sel).lead_time.values.astype('timedelta64[h]').astype(int)
            idx = np.where((dts2 <= max(dts))&(dts2 != 0))[0]
            rms2 = rms2[idx] ; dts2 = dts2[idx]
            # if 'climatology' in cmodel:
            #     rms2 = np.append(rms2, rms2[-1]*(len(dts)-len(rms2)))
            if cmodel == "hres":
                norm = rms2, dts2
            #print("hey norm is", norm, "rms2", rms2.shape, norm[0].shape)
            if NORM: ax.plot(dts2,(rms2-norm[0])/norm[0]*100,label=cmodel)
            else: ax.plot(dts2,rms2,label=cmodel)
            ax.set_title(v)
            ax.grid(1)

        for modelname in ['neoquadripede_331M', 
                          #'ultralatentbachelor_168M',
                          'joanlatentsucks'
                          #'rtyblong',
                          ]:
            def get_avg_rms(dt):
                s = get_model_scores(modelname,v,dt=dt,metric=metric)
                if s[0] is None: return None
                rms = np.array(sorted(s[1]))
                rms = rms[3:-3] # remove few outliers
                return np.mean(np.square(rms))**0.5
            
            rms = []
            dts_valid = []
            for dt in dts:
                rms_dt = get_avg_rms(dt)
                if rms_dt is None: continue
                rms.append(rms_dt)
                dts_valid.append(dt)
            rms = np.array(rms)
            dts_valid = np.array(dts_valid)
            if '133_q' in v: rms /= 1e6

            if NORM:
                try:
                    idx = [list(norm[1]).index(a) for a in dts_valid]
                except:
                    print(f"No norm found for {modelname} {v} {metric}")
                    idx = [0]*len(dts_valid)
                #print("uh oh",rms.shape, dts, norm[1][idx])
                ax.plot(dts_valid,(rms-norm[0][idx])/norm[0][idx]*100,label=modelname)
            else: ax.plot(dts_valid,rms,label=modelname)
        ax.legend()

    fig.tight_layout()
    os.makedirs('ignored',exist_ok=True)
    fn = f'ignored/3x3{metric}.png'
    plt.savefig(fn)
    print(f"Saved to {fn}")


# download weatherbench data eg.
# gsutil -m cp -r gs://weatherbench2/results/1440x721/deterministic /fast/weatherbench2_results/ 