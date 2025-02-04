from datetime import datetime, timedelta, timezone
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
sys.path.append('/fast/wbhaoxing/deep')
from utils import *
from data import *
from eval import *


from evals.evaluate_error import *
from evals.package_neo import *

globals().update(DEFAULT_CONFIG.__dict__)

S3_UPLOAD = True

TARGET_FORECAST_HOUR = 21*24 + 6
FORECAST_HOUR_STEP = 6
S3_CLIENT = boto3.client('s3')
BUCKET = 'wb-dlnwp'
    

model = None
device = None
# Separate thread for uploading to S3 to avoid bottlenecking the main thread
upload_queue = queue.Queue()
upload_thread = None

def kill_prev_process():
    mypid = os.getpid()
    cmd = """nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} sh -c \'ps -fp {} | grep "run_rt_det"  | awk "{print \\$2}"\' | grep -vw """ + str(mypid)
    print(cmd)
    pids_to_del = subprocess.getoutput(cmd).replace('\n', ' ')
    if len(pids_to_del) > 0:
        print(f"Killing PIDs {pids_to_del} that are still running run_rt_det.py and keeping all the GPU RAM to themselves")
        subprocess.Popen(f"kill -9 {pids_to_del}", shell=True)

def load_model():
    # we store it as global, but only load it when we need it
    # this lets booting happen slightly faster so if things are crashing lots we don't get as wrecked
    global model
    global device

    if model is not None:
        return
    
    getter = globals().get(f"get_{MODEL_NAME}")
    if getter is None:
        raise ValueError(f"get_{MODEL_NAME}() not found in evals.package_neo")
    model = getter()

    device = torch.device("cuda:0")
    kill_prev_process()
    model = model.to(device)
    model.eval()


def latest_hres_date():
    latest_date = None

    for file in os.listdir(f"/fast/proc/hres_rt/f000/"):
        if not file.endswith(".npz"):
            continue

        date = datetime.strptime(file[:-4], "%Y%m%d%H")
        if latest_date is None or date > latest_date:
            latest_date = date

    return latest_date


def latest_gfs_date():
    last_timestamp = None

    for file in os.listdir(f"/fast/proc/gfs_rt/f000"):
        if not file.endswith(".npz"):
            continue

        timestamp = int(file.split(".")[0])
        if last_timestamp is None or timestamp > last_timestamp:
            last_timestamp = timestamp

    date = datetime.utcfromtimestamp(last_timestamp)
    return date


def check_processed(date):
    nsteps = TARGET_FORECAST_HOUR // FORECAST_HOUR_STEP
    hr = 0
    processed_count = 0

    for step in range(nsteps):
        hr += FORECAST_HOUR_STEP
        output_path = f'{OUTPUT_PATH}/{to_filename(date, hr)}.*.npy'
        if glob.glob(output_path):
            processed_count += 1

    return processed_count == nsteps


def run_and_save(date, rollout_schedule=None, min_dt=1):
    """
    Runs for a given date
    """

    with torch.no_grad():
        print(model.config.inputs)

        model.config.inputs[1].source = "hres_rt-13"
        model.config.inputs[1].input_levels = levels_hres
        model.config.inputs[1].intermediate_levels = [levels_tiny]

        dataset = WeatherDataset(
            DataConfig(
                inputs=model.config.inputs,
                outputs=model.config.inputs,  # the output format. Set ot be identical to the input format
                timesteps=[0],  # timesteps to load for output eval, set to 0 to make this happy
                requested_dates=[date],
                only_at_z=[0,6,12,18],  # The hours for each date
                clamp_output=np.inf,  # Do not bound the output; this is only used in training
                realtime=True,
            )
        )

        dataset.check_for_dates()

        sample = default_collate([dataset[0]])
        x = sample[0]
        
        def save_hour(ddt, y):
            y = y.to('cuda')
            xu,y = unnorm_output(x[0],y,model,ddt, y_is_deltas=False, skip_x=True, pad_x=True)
        
            h = round(ddt)
            save_upload_output(y.to("cpu").numpy(), h, date)


        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            x = [xx.to('cuda') for xx in x]
            dts = list(range(FORECAST_HOUR_STEP, TARGET_FORECAST_HOUR, FORECAST_HOUR_STEP))
            dts = list(range(0, 24*5, 1)) + list(range(24*5, TARGET_FORECAST_HOUR, FORECAST_HOUR_STEP))
            print(f"Generating forecasts for timesteps: {dts}", flush=True)
            ys = model.forward(x, dts, send_to_cpu=True, callback=save_hour)

def upload_worker():
    while True:
        task = upload_queue.get()
        if task is None:
            break
        output_fn, h = task
        upload_output(output_fn, h)
        upload_queue.task_done()

def save_upload_output(out, h, date):
    output_fn = save_output(out, h, date)
    if S3_UPLOAD:
        upload_queue.put((output_fn, h))

def save_output(out, h, date):
    print(f"Saving forecast hour {h}", flush=True)
    start_time = time.time()
    # Save output filename (including tags added by save_instance)
    output_fn = save_instance(out, f'{OUTPUT_PATH}/{get_output_filepath(date, round(h))}', model.config.output_mesh, MODEL_NAME, upload_metadata=S3_UPLOAD)[len(OUTPUT_PATH)+1:]
    print(f"Saved fh {h} ({round((time.time() - start_time) * 1000)} ms) to {output_fn}", flush=True)
    gc.collect()
    return output_fn

def upload_output(output_fn, h):
    print(f"Uploading forecast hour {h} to s3", flush=True)
    start_time = time.time()
    s3_fn = f'{MODEL_NAME}/{output_fn}'.replace('+', '/det/', 1)
    S3_CLIENT.upload_file(f'{OUTPUT_PATH}/{output_fn}', BUCKET, s3_fn)
    print(f"Uploaded fh {h} to s3 bucket {BUCKET} at {s3_fn} ({round((time.time() - start_time) * 1000)} ms)", flush=True)

def run_if_needed(idempotent=True, forecast_zero=None, rollout_schedule=None, min_dt=1):
    if forecast_zero is None:
        forecast_zero = latest_hres_date()
        if latest_gfs_date() < forecast_zero:
            forecast_zero = latest_gfs_date()

        print(f"Latest data: {forecast_zero} (latest GFS: {latest_gfs_date()}; latest HRES: {latest_hres_date()})", flush=True)
    else:
        if isinstance(forecast_zero, str):
            forecast_zero = datetime.strptime(forecast_zero, "%Y%m%d%H").replace(tzinfo=timezone.utc)

        print(f"Forecast zero: {forecast_zero} ({forecast_zero.timestamp()})", flush=True)

    if check_processed(forecast_zero) and idempotent:
        print("Already processed", flush=True)
        return

    load_model()
    if S3_UPLOAD:
        upload_thread = threading.Thread(target=upload_worker)
        upload_thread.start()
        run_and_save(forecast_zero, rollout_schedule=rollout_schedule, min_dt=min_dt)
        upload_queue.put(None)  # Stop the worker
        upload_thread.join()  # Wait for the worker to finish
    else:
        run_and_save(forecast_zero, rollout_schedule=rollout_schedule, min_dt=min_dt)


def main():
    print(f"Started process {os.getpid()} for run_beta_det", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('forecast_zero', nargs='?', help='Forecast zero time in YYYYMMDDHH format')
    parser.add_argument('--model', type=str)
    parser.add_argument('--full-time-res', action='store_true')  # handled as global
    parser.add_argument('--idempotent', action='store_true')
    parser.add_argument('--min-dt', type=int, default=1)
    parser.add_argument('--max-forecast-hour', type=int)
    parser.add_argument('--no-upload', action='store_true')

    args = parser.parse_args()

    assert args.model is not None, "--model must be specified"
    global OUTPUT_PATH
    global MODEL_NAME
    # force model name to be specified in beta dag so this script doesn't need to be modified
    OUTPUT_PATH = f"/fast/realtime/outputs/{args.model}"
    MODEL_NAME = args.model

    if args.max_forecast_hour:
        global TARGET_FORECAST_HOUR
        TARGET_FORECAST_HOUR = args.max_forecast_hour

    if args.min_dt > 1:
        global FORECAST_HOUR_STEP
        FORECAST_HOUR_STEP = args.min_dt
    if args.no_upload:
        global S3_UPLOAD
        S3_UPLOAD = False
    run_if_needed(forecast_zero=args.forecast_zero, idempotent=args.idempotent, min_dt=args.min_dt)


if __name__ == '__main__':
    main()
