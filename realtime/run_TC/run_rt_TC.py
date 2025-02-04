from datetime import datetime, timezone
import time
import sys
import numpy as np
import os
import glob
import gc
import argparse
import boto3
import threading
import queue
import subprocess

from utils import *
from data import *
from eval import *

from evals.evaluate_error import *
from evals.package_neo import *
from realtime.consts import DEFAULT_MODEL_NAME

class RunTCModel(): 
    def __init__(self, model_string, dts, forecast_zero, gpu_id=0, output_base_path='/huge/users/jack/evals/'):
        self.model_string = model_string
        self.model_function = 'get_' + model_string
        self.output_path = output_base_path + model_string
        self.dts = dts
        self.forecast_zero = forecast_zero
        self.device = torch.device(f"cuda:{gpu_id}")
        
    def run(self):
        self.load_model()
        self.run_and_save(self.forecast_zero)
    
    def load_model(self):
        getter = globals().get(self.model_function)
        if getter is None:
            raise ValueError(f"{self.model_function}() not found in evals.package_neo")
        self.model = getter()
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def save_output(self, out, h, date):
        print(f"Saving forecast hour {h}", flush=True)
        start_time = time.time()
        # Save output filename (including tags added by save_instance)
        output_fn = save_instance(out, f'{self.output_path}/{get_output_filepath(date, round(h))}', self.model.config.outputs[0], self.model_string, upload_metadata=False, )[len(self.output_path)+1:]
        print(f"Saved fh {h} to {output_fn} ({round((time.time() - start_time) * 1000)} ms)", flush=True)
        gc.collect()
        
    def save_intensity(self, out, h, date):
        print(f"Saving intensity for forecast hour {h}", flush=True)
        start_time = time.time()
        _, hash = self.model.config.outputs[0].to_json(self.model_string)
        path = f'{self.output_path}/{get_output_filepath(date, round(h))}'
        np.save(path + f"_intensity.{hash}.npy", out)
        end_time = time.time()
        print(f"Saved intensity for fh {h} to {path} ({round((end_time - start_time) * 1000)} ms)", flush=True)
    
    def run_and_save(self, date):
        with torch.no_grad():
            print(self.model.config.inputs)

            dataset = WeatherDataset(
                DataConfig(
                    inputs=self.model.config.inputs,
                    outputs=self.model.config.inputs,  # the output format. Set ot be identical to the input format
                    timesteps=[0],  # timesteps to load for output eval, set to 0 to make this happy
                    requested_dates=[date],
                    only_at_z=list(range(0, 24, 3)),  # The hours for each date
                    clamp_output=np.inf,  # Do not bound the output; this is only used in training
                    realtime=True,
                )
            )

            dataset.check_for_dates()

            from train import collate_fn
            sample = collate_fn([dataset[0]])
            x = sample.get_x_t0()

            def save_hour(ddt, y_list):
                for i, y in enumerate(y_list):
                    if i == 0:
                        y = y.to('cuda')
                        xu, y = unnorm_output(x[0], y, self.model, ddt, y_is_deltas=False, skip_x=True, pad_x=True)
                        h = round(ddt)
                        self.save_output(y.to("cpu").numpy(), h, date)
                    else:
                        self.save_intensity(y.to("cpu").numpy(), h, date)

            with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
                x = [xx.to('cuda') for xx in x]
                print(f"Generating forecasts for timesteps: {self.dts}", flush=True)
                ys = self.model.forward(x, self.dts, send_to_cpu=True, callback=save_hour)
        
model_names = [
    #'TCregional',
    'TCregionalio',
    #'bachelor',
    #'TCfullforce',
    #'TCvarweight',
]

def main():
    # Runs like:
    # python3 realtime/run_rt_TC.py YYYYMMDDHH [--flags]
    print(f"Started process {os.getpid()} for run_rt_TC", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('forecast_zero', nargs='?', help='Forecast zero time in YYYYMMDDHH format')
    parser.add_argument('--dts', type=str, help='Comma-separated list of forecast hours to run')
    parser.add_argument('--dts_up_to', type=int, help='Run all forecast hours up to this value (in days)')
    args = parser.parse_args()

    if not args.dts and not args.dts_up_to:
        raise(ValueError("Must specify either --dts or --dts_up_to"))

    if args.dts:
        dts = [int(dt) for dt in args.dts.split(',')]

    if args.dts_up_to:
        dts = list(range(0, (args.dts_up_to * 24) + 1, 6))

    if isinstance(args.forecast_zero, str):
        forecast_zero = datetime.strptime(args.forecast_zero, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    else:
        raise(ValueError("forecast_zero must be a string"))

    for model_name in model_names:
        print("--------------------------------------", flush=True)
        print(f"Running model {model_name}", flush=True)
        start = time.time()
        model = RunTCModel(model_name, dts, forecast_zero) # Defaults to gpu 0
        model.run()
        print(f"Finished running model {model_name}, took {time.time() - start} seconds", flush=True)
        print("😎😎😎 Waiting for 10 seconds before running the next model 😎😎😎", flush=True)
        print("--------------------------------------", flush=True)
        time.sleep(10)

if __name__ == '__main__':
    main()