import json
import pytz
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d as gf
import os
from datetime import datetime, timezone
import sys
sys.path.append('../../')
from eval import all_metric_fns
from evals import *
from utils import *
from collections import defaultdict
from evals.s2s.s2s_eval_ifs import s2s_var_mapping

cache = {}

def get_s2s_model_scores(model,dt=24,vs='era5_daily-28',metric='rmse', resolution=1.5, start_date=None, end_date=None):
    global cache
    dates = []
    vars = defaultdict(list)
    for fn in sorted(os.listdir(get_errors_path(model, resolution=resolution))):
        fns = fn.split('.')
        if len(fns) < 3: continue
        if not fns[0].endswith(f'+{dt}') or fns[1] != 'vs_'+vs or fns[2] != 'json':
            continue
        pp = get_errors_path(model, resolution=resolution)+fn
        if pp in cache:
            j = cache[pp]
        else:
            with open(pp,'r') as f:
                j = json.load(f)
            cache[pp] = j
        rms = j[metric]
        date = datetime.strptime(fn.split('+')[0],'%Y%m%d%H')#.replace(tzinfo=timezone.utc)
        if start_date is not None and date < start_date: continue
        if end_date is not None and date > end_date: continue
        dates.append(date)
        for k in rms:
            vars[k].append(rms[k])
    assert len(dates) > 0, f"No dates found for {model} {dt} {resolution}"
    assert len(set([len(vars[k]) for k in vars])) == 1, f"Different # dates per variable for {model} {dt} {resolution}"
    #dates, vars = zip(*sorted(zip(dates, vars)))
    return dates, vars

metric = 'rmse'
NDAY = 45
dts = np.arange(0,NDAY+1)*24
grid_resolution = 1.5
start_date = datetime(2020,1,1)
start_date_str = start_date.strftime("%Y%m%d")
end_date = datetime(2020,12,31)
end_date_str = end_date.strftime("%Y%m%d")
plot_vars = list(s2s_var_mapping.values()) + ["165_10u", "166_10v", "151_msl", "179_ttr"]
modelnames = ['freyja', 'tyr']

for modelname in modelnames:
    errors_proc_dir = f'/huge/deep/evaluation/{modelname}/errors_proc/{grid_resolution}deg/{start_date_str}-{end_date_str}/'#{v}.json'
    #if 'errors_proc/{grid_resolution}deg/{start_date}-{end_date}/{v}.json'
    if not 0: # os.path.exists(errors_proc_path):
        os.makedirs(errors_proc_dir, exist_ok=True)
        def get_avg_rms(dt):
            dates, errors_by_var = get_s2s_model_scores(modelname,dt=dt,metric=metric, resolution=grid_resolution, start_date=start_date, end_date=end_date)
            rms_by_v = {}
            for v, errors in errors_by_var.items():
                rms = np.array(sorted(errors))
                rms = rms[3:-3] # remove few outliers
                rms_by_v[v] = float(all_metric_fns[metric](rms))
            return rms_by_v
        
        rms_by_v_dt = defaultdict(list)
        for dt in dts:
            rmses = get_avg_rms(dt)
            for v, rms in rmses.items():
                rms_by_v_dt[v].append(rms)
        out = {}
        out['start_date'] = start_date_str
        out['end_date'] = end_date_str
        out['grid_resolution'] = grid_resolution
        out['model_type'] = modelname
        out['comparison_source'] = 'era5_daily'
        for v, rms in rms_by_v_dt.items():
            out['variable'] = v
            out['rmse'] = {str(dt): rms_by_v_dt[v][i] for i, dt in enumerate(dts)}
            errors_proc_path = errors_proc_dir+f'{v}.json'
            with open(errors_proc_path, 'w') as f:
                json.dump(out, f)
    
fig, axs = plt.subplots(3, 3,figsize=(15,12))
dts_days = dts/24
for i, v in enumerate(plot_vars):
    ax = axs.flatten()[i]
    ecmwf_errors_path = f'/huge/proc/s2s/ecmf/rt/daily/errors_proc/{start_date_str}-{end_date_str}/{v}.json'
    if os.path.exists(ecmwf_errors_path):
        with open(ecmwf_errors_path, 'r') as f:
            data = json.load(f)
        ecmwf_rms = np.array([data['rmse'][str(dt)] for dt in dts])
        ax.plot(dts_days, ecmwf_rms, label='ecmwf', color=f'C0')
    for j, modelname in enumerate(modelnames):
        if modelname == 'freyja' and v == '136_tcw':
            continue
        errors_proc_path = f'/huge/deep/evaluation/{modelname}/errors_proc/{grid_resolution}deg/{start_date_str}-{end_date_str}/{v}.json'
        if not os.path.exists(errors_proc_path):
            continue
        with open(errors_proc_path, 'r') as f:
            data = json.load(f)
        model_rms = np.array([data['rmse'][str(dt)] for dt in dts])
        ax.plot(dts_days, model_rms, label=modelname, color=f'C{j+1}')
    ax.set_title(v)
    ax.grid(1)
    ax.legend()

fig.suptitle(f'{metric} vs forecast lead time (in days) @ {grid_resolution}deg grid')

fig.tight_layout()
os.makedirs('ignored',exist_ok=True)
fn = f'ignored/s2s_{metric}_{grid_resolution}deg.png'
plt.savefig(fn)
print(f"Saved to {fn}")

