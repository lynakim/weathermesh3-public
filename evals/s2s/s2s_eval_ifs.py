import os
from dateutil.rrule import rrule, MONTHLY

import pygrib
import numpy as np
from datetime import datetime, timedelta

import torch
from tqdm.auto import tqdm
import sys
sys.path.append('../../')
from utils import core_sfc_vars, log_vars
from eval import all_metric_fns
import pickle
import json

ecmf_model_dir = '/huge/proc/s2s/ecmf/rt/daily'
s2s_var_mapping = {
        '2 metre temperature': '167_2t',
        '2 metre dewpoint temperature': '168_2d',
        #'Sea surface temperature': '034_sstk', # need to account for nans
        'Total column water': '136_tcw',
        'Total Cloud Cover': '45_tcc',
        # {'2 metre temperature', '2 metre dewpoint temperature', 'Sea surface temperature', 'Total Cloud Cover', 'Soil moisture top 100 cm', 'Snow density', 'Sea ice area fraction', 'Snow depth water equivalent', 'Soil temperature top 20 cm', 'Snow albedo', 'Skin temperature', 'Soil temperature top 100 cm', 'Soil moisture top 20 cm', 'Total column water', 'Convective available potential energy'}
    }

def eval_metrics_sfc(preds, actuals, weights, metric_fns=['rmse', 'bias']):
        assert preds.shape == actuals.shape, f"{preds.shape} != {actuals.shape}"
        #assert preds.shape[0] == weights.shape, f"preds and weights must have same spacial dims. preds: {preds.shape}, weight: {weight.shape}"
        
        error = preds - actuals
        error = torch.Tensor(error)
        weights = torch.Tensor(weights)
        metrics = {}
        
        for name in metric_fns:
            rmsall = all_metric_fns[name](error,weights)
            metrics[name] = float(rmsall)
        return metrics


def load_era5_daily(variable, timestamp):
    """Load ERA5 daily data for a specific variable and timestamp."""
    date_str = timestamp.strftime("%Y%m%d")
    year_str = timestamp.strftime("%Y")
    
    if variable in core_sfc_vars: 
        path = f'/fast/proc/era5_daily/f000/{year_str}/{date_str}.npz'
        data = np.load(path)['sfc']
        data = data[:, :, core_sfc_vars.index(variable)]  
    else:  # extra surface variables
        path = f'/fast/proc/era5_daily/extra/{variable}/{year_str}/{date_str}.npz'
        data = np.load(path)['x']
        data = data[0] if data.ndim == 3 else data
    with open(f"/fast/consts/normalization.pickle", "rb") as f:
        norm = pickle.load(f)
        mean, std = norm[variable][0], np.sqrt(norm[variable][1])
    data = data * std + mean
    if variable in log_vars:
        data = np.exp(data)
    return data

def create_forecast_jsons(s2s_grib_path):
    grbs = pygrib.open(s2s_grib_path)
    
    results = {}
    lats = np.arange(90, -90, -1.5) 
    lats = np.repeat(lats, 240).reshape(120,240)
    weights = np.cos(lats * np.pi/180) 
    metric_names = ['rmse', 'bias']

    # Loop through messages in the grib file
    for grb in grbs:
        var_name = grb.name
        if var_name not in s2s_var_mapping:
            continue
        era5_var = s2s_var_mapping[var_name]
        date = datetime.strptime(str(grb.dataDate), '%Y%m%d')
        step = grb.forecastTime # 24 means 24-48h period
        forecast_date = date + timedelta(hours=step)
        out = {}
        out['input_date'] = str(grb.dataDate)
        out['output_date'] = forecast_date.strftime('%Y%m%d')
        out['forecast_dt_hours'] = step
        out['model_type'] = 'ecmf'
        out['comparison_source'] = 'era5_daily-0'
        out['grid_resolution'] = '1.5'
        out['variable'] = var_name
        path = f'{ecmf_model_dir}/errors/{out["grid_resolution"]}deg/{era5_var}/{str(grb.dataDate)}+{step}.vs_era5_daily-0.json'
        if os.path.exists(path):
            print(f"Skipping {path} as it already exists")
            continue
        # Get S2S data
        s2s_data = grb.values # 1.5 degree grid
        assert s2s_data.shape == (121,240), f"S2S data shape {s2s_data.shape} does not match expected (121,240)"
        s2s_data = s2s_data[:120,:] # remove last row in solidarity with ERA5
        if var_name == 'Total Cloud Cover':
            s2s_data = s2s_data/100 # convert from % to fraction
            
        try:
            era5_data = load_era5_daily(era5_var, forecast_date) # 0.25 degree grid
        except FileNotFoundError:
            print(f"Missing ERA5 data for {forecast_date}")
            continue
        assert era5_data.shape == (720,1440), f"ERA5 data shape {era5_data.shape} does not match expected (720,1440)"
        
        era5_downsampled = era5_data[::6,::6]
        assert era5_downsampled.shape == (120,240), f"ERA5 downsampled data shape {era5_downsampled.shape} does not match expected (120,240)"
        # TODO: handle vars with nans, union of nans from masks
        metrics = eval_metrics_sfc(s2s_data, era5_downsampled, weights, metric_names)
        
        for metric_name in metric_names:
            out[metric_name] = metrics[metric_name]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,'w') as f:
            json.dump(out,f,indent=2)
        print(f"Saved {path}")            

def main():
    # 1. create all forecast jsons
    start_month = "202001"
    end_month = "202012"
    start_date = datetime.strptime(start_month, '%Y%m')
    end_date = datetime.strptime(end_month, '%Y%m')
    for date in tqdm(list(rrule(MONTHLY, dtstart=start_date, until=end_date))):
        s2s_grib_path = f'{ecmf_model_dir}/{date.strftime("%Y%m")}.grib'
        create_forecast_jsons(s2s_grib_path)
        print(f"Processed month {date}")

    # 2. read all forecast jsons per variable and calculate overall metrics by lead time
    # path = f'{ecmf_model_dir}/errors/{out["grid_resolution"]}deg/{era5_var}/{str(grb.dataDate)}+{step}.vs_era5_daily-0.json'
    # start_date = datetime(2020,1,1)
    # end_date = datetime(2020,12,31)
    # for variable, era5_var in s2s_var_mapping.items():
    #     path = f'{ecmf_model_dir}/errors/1.5deg/{era5_var}/'
    #     metrics = {'rmse': {}, 'bias': {}}
    #     # list all files in path
    #     for file in os.listdir(path):
    #         init_date, step = file.split('+')
    #         init_date = datetime.strptime(init_date.split('/')[-1], '%Y%m%d')
    #         step = int(step.split('.')[0])
    #         forecast_date = init_date + timedelta(hours=step)
    #         if init_date < start_date or forecast_date > end_date: # this takes init_dates right up to end_date, while evaluate_errs_neo.py might stop at end_date - forecast_dt
    #             continue            
    #         with open(os.path.join(path, file), 'r') as f:
    #             data = json.load(f)
    #         assert data['variable'] == variable, f"Variable mismatch: {data['variable']} != {variable}"
    #         assert data['forecast_dt_hours'] == step, f"Forecast dt mismatch: {data['forecast_dt_hours']} != {step}"
    #         for metric_name in metrics.keys():
    #             assert metric_name in data.keys(), f"Metric {metric_name} not found in {file}"
    #             if step not in metrics[metric_name].keys():
    #                 metrics[metric_name][step] = []
    #             metrics[metric_name][step].append(data[metric_name])
    #         print(f"Processed {file}")
    #     print(f"Processed {variable}")
    
    #     for metric_name in metrics.keys():
    #         for step in metrics[metric_name].keys():
    #             metric_fn = all_metric_fns[metric_name]
    #             metrics[metric_name][step] = float(metric_fn(torch.Tensor(metrics[metric_name][step])))
    #     metrics['variable'] = variable
    #     metrics['start_date'] = start_date.strftime("%Y%m%d")
    #     metrics['end_date'] = end_date.strftime("%Y%m%d")
    #     metrics['grid_resolution'] = data['grid_resolution']
    #     metrics['model_type'] = data['model_type']
    #     metrics['comparison_source'] = data['comparison_source']
        
    #     error_proc_path = f'{ecmf_model_dir}/errors_proc/{start_date.strftime("%Y%m%d")}-{end_date.strftime("%Y%m%d")}/{era5_var}.json'
    #     os.makedirs(os.path.dirname(error_proc_path), exist_ok=True)
    #     with open(error_proc_path, 'w') as f:
    #         json.dump(metrics, f, indent=2)
    #     print(f"Saved {variable} metrics to file")
    
    # print(metrics)

if __name__ == "__main__":
    main()