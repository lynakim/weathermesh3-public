from datetime import datetime, timezone
import time
import sys
import numpy as np
import os
import glob
import gc
import traceback
import subprocess
import argparse
import pathlib

from utils import *
from data import *
from eval import *
from evals.evaluate_error import *
from evals.package_neo import *
from realtime.consts import DEFAULT_MODEL_NAME, DEFAULT_MODEL_HASH
import boto3
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
globals().update(DEFAULT_CONFIG.__dict__)

OUTPUT_PATH, S3_OUTPUT_PATH = None, None
MODEL_NAME = 'rtyamahabachelor5'
IC_INPUT_SOURCE_PATH = '/huge/users/criedel/DAOutput/UPDATED-ICs'
IC_INPUT_PATH = '/fast/realtime/inputs/WeatherMeshDA'
ENS_HOURS_TO_SAVE = list(range(0,24*7,6)) + list(range(24*7, 24*15+1, 24)) # for now for DA members only save daily other than the first 6-hour step
DA_HOURS_TO_SAVE = [0,6] + [24 * i for i in range(1, 8)] # for now for DA members only save daily other than the first 6-hour step
HASH = DEFAULT_MODEL_HASH

ENSEMBLE_COUNT = 51 # the ECMWF number
RUN_MEMBERS = range(25) # 0 is the control member
#MEMBERS_ONLY = '--members-only' in sys.argv

OVERWRITE_IFS_INPUT_PATH = None
# test 24h forecast for member 0
# ENS_HOURS_TO_SAVE = [24]
# RUN_MEMBERS = [0]
# OVERWRITE_IFS_INPUT_PATH = '/huge/users/anuj'

S3_INPUT_PATH = 'WeatherMeshDA'
S3_CLIENT = boto3.client('s3')
BUCKET = 'wb-dlnwp'

NUM_UPLOAD_THREADS = 16
NUM_ENSEMBLE_THREADS = 2

model = None
device = None
# Separate thread for uploading to S3 to avoid bottlenecking the main thread
upload_queue = queue.Queue()
upload_thread = None

def kill_prev_process():
    mypid = os.getpid()
    
    # cmd = """nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} sh -c \'ps -fp {} | grep "run_rt_ens"  | awk "{print \\$2}"\' | grep -vw """ + str(mypid)
    cuda_devices = os.environ.get('CUDA_AVAILABLE_DEVICES')
    
    # If not set or empty, assume GPU 0
    target_gpu = '0'
    if cuda_devices:
        try:
            target_gpu = cuda_devices.split(',')[0].strip()
        except IndexError:
            target_gpu = '0'
    
    # Get processes on target GPU
    cmd = f"""nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | 
             grep -w "$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader -i {target_gpu})" |
             awk -F', ' '{{{{print $2}}}}' |
             xargs -I {{}} sh -c 'ps -fp {{}} | grep "run_rt_ens" | awk "{{print \\$2}}"' |
             grep -vw {mypid}"""
    
    print(cmd)
    pids_to_del = subprocess.getoutput(cmd).replace('\n', ' ')
    if len(pids_to_del) > 0:
        print(f"Killing PIDs {pids_to_del} that are still running run_rt_ens.py and keeping all the GPU RAM to themselves")
        subprocess.Popen(f"kill -9 {pids_to_del}", shell=True)

def load_model():
    # we store it as global, but only load it when we need it
    # this lets booting happen slightly faster so if things are crashing lots we don't get as wrecked
    global model
    global device

    if model is not None:
        return

    # from gen1.package import get_hegelquad # till 2024-10-03
    
    getter = globals().get(f"get_{MODEL_NAME}")
    if getter is None:
        raise ValueError(f"get_{MODEL_NAME}() not found in evals.package_neo")
    model = getter()
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()


def latest_ens_date():
    by_date = {}

    for ens_dir in os.listdir("/fast/proc/ens_rt/"):
        if not ens_dir.isdigit():
            continue

        ens = int(ens_dir)
        for file in os.listdir(f"/fast/proc/ens_rt/{ens_dir}/f000/"):
            if not file.endswith(".npz"):
                continue

            date = datetime.strptime(file[:-4], "%Y%m%d%H")
            if date not in by_date:
                by_date[date] = set()

            by_date[date].add(ens)

    for date, ensembles in sorted(by_date.items(), reverse=True):
        if len(ensembles) >= ENSEMBLE_COUNT:
            return date


def latest_gfs_date():
    last_timestamp = None

    for file in os.listdir("/fast/proc/gfs_rt/f000"):
        if not file.endswith(".npz"):
            continue

        timestamp = int(file.split(".")[0])
        if last_timestamp is None or timestamp > last_timestamp:
            last_timestamp = timestamp

    date = datetime.utcfromtimestamp(last_timestamp)
    return date


def check_processed_member(date, member):
    steps = ENS_HOURS_TO_SAVE
    processed_count = 0

    for hr in steps:# check s3 and local
        s3_path = f'{S3_OUTPUT_PATH}/{get_date_str(date)}/ens{member}/{hr}.'
        # pattern match the hash in the filename
        if (S3_OUTPUT_PATH is not None and S3_CLIENT.list_objects_v2(Bucket=BUCKET, Prefix=s3_path)['KeyCount']) or (S3_OUTPUT_PATH is None and glob.glob(f'{OUTPUT_PATH}/{get_date_str(date)}/ens{member}/{hr}.{HASH}.npy')):
            processed_count += 1
    return processed_count == len(steps)    

def dispatch_ensemble_viz(ens):
    if '--no-viz' in sys.argv:
        print("Skipping ensemble viz dispatch", flush=True)
        return

    try:
        model_name = OUTPUT_PATH.split('/')[-1]
        command = f"cd /home/ubuntu/dlviz && /usr/bin/python3 process_dl_output.py {model_name}_{ens}"
        print(f"Running `{command}`", flush=True)
        subprocess.Popen(command, shell=True)
    except Exception as e:
        print(f"Failed to dispatch ensemble viz for {ens}: {e}", flush=True)
        traceback.print_exc()

def normalize_input(x: np.ndarray, mesh) -> np.ndarray:
    start_shape = x.shape
    print('x.shape', x.shape)
    with open(f"{CONSTS_PATH}/normalization.pickle", "rb") as f:
        norm = pickle.load(f)

    pressure_vars = mesh.pressure_vars
    sfc_vars = mesh.sfc_vars

    # shapes
    # (720, 1440, 5, 28) (720, 1440, 4)
    x = x.squeeze()
    n_levels = mesh.n_levels
    x_pr, x_sfc = x[:,:,:5*n_levels], x[:,:,5*n_levels:]
    x_pr = np.reshape(x_pr, (720, 1440, 5, n_levels))
    print("shapes (x_pr, x_sfc): ", x_pr.shape, x_sfc.shape)

    mean_sfc_arr, std_sfc_arr = np.empty_like(x_sfc), np.empty_like(x_sfc)
    for i, var in enumerate(sfc_vars):
        if var == "zeropad":
            continue
        mean, norm_var = norm[var]
        mean_sfc_arr[:,:,i] = mean # shape (720, 1440, 4)
        std_sfc_arr[:,:,i] = np.sqrt(norm_var) # shape (720, 1440, 4)
    x_sfc_normed = (x_sfc - mean_sfc_arr) / std_sfc_arr

    wh_lev = mesh.wh_lev
    mean_pr_arr, std_pr_arr = np.empty_like(x_pr), np.empty_like(x_pr) # (720, 1440, 5, 28)
    for i, var in enumerate(pressure_vars):
        mean, norm_var = norm[var]
        mean_pr_arr[:,:,i,:] = mean[wh_lev] # shape (720, 1440, 5, 28)
        std_pr_arr[:,:,i,:] = np.sqrt(norm_var)[wh_lev] # shape (720, 1440, 5, 28)
    x_pr_normed = (x_pr - mean_pr_arr) / std_pr_arr

    if mesh.extra_sfc_pad > 0:
        # make sure the extra sfc pads are zeroed out
        x_sfc_normed[:,:,-mesh.extra_sfc_pad:] = 0

    # check magnitude now
    print("Norms (sfc, pr): ", np.abs(x_sfc_normed).mean(), np.abs(x_pr_normed).mean())

    x = np.concatenate([x_pr_normed.reshape(720, 1440, -1), x_sfc_normed], axis=-1)
    assert x.shape == start_shape
    return x

def process_input(file_path: str, date: datetime, mesh):
    x = np.load(file_path)
    print(f"np.abs(x).mean(): {np.abs(x).mean()}")
    x = normalize_input(x, mesh)
    x = torch.from_numpy(x).unsqueeze(0)
    # it appears the model expects a list of two tensors,
    # with the first being the IC and the second being a 
    # (1,) tensor of the timestamp
    x_ts = [x, torch.tensor([date.replace(tzinfo=timezone.utc).timestamp()])]
    return x_ts

def get_ic_relative_path(date: str, ens: int):
    return f"{date}/ens{ens}/0.{HASH}.npy"

def run_member(date: datetime, ens, use_da=False, use_huge=False, idempotent=True):
    """
    Runs an ensemble member for a given date
    ens is the ensemble number
    time_horizon is the time horizon to which to rollout model
    hours_to_save is None if saving output at all hours (so it's a bit of a misnomer sry)
        it is a list of hours at which to save otherwise
    """
    
    if idempotent and check_processed_member(date, ens):
        print(f"Skipping member {ens + 1}/{ENSEMBLE_COUNT}", flush=True)
        return 1
    
    print(f"Running member {ens + 1}/{ENSEMBLE_COUNT}", flush=True)

    def save_hour(ddt, y):
        y = y.to('cuda')
        xu,y = unnorm_output(x[0],y,model,ddt, y_is_deltas=False, skip_x=True, pad_x=True)
    
        h = round(ddt)
        save_upload_output(y.to("cpu").numpy(), h, date, str(ens))

    with torch.no_grad():
        todo = ENS_HOURS_TO_SAVE
        if not use_da: # run on data as normal
            inputs = copy.deepcopy(model.config.inputs) # else threads will silently use the wrong ens member T_T
            if OVERWRITE_IFS_INPUT_PATH is not None:
                inputs[1].load_locations = [OVERWRITE_IFS_INPUT_PATH] # to load ifs ens inputs from a different location
            dataset = WeatherDataset(
                DataConfig(
                    inputs=inputs,
                    outputs=inputs,  # the output format. Set ot be identical to the input format
                    timesteps=[0],  # timesteps to load for output eval, set to 0 to make this happy
                    requested_dates=[date],
                    only_at_z=[0],  # The hours for each date
                    clamp_output=np.inf,  # Do not bound the output; this is only used in training
                    realtime=True,
                    ens_nums=[None, ens],  # [gfs, ecmwf], and gfs has no ensemble number
                )
            )

            dataset.check_for_dates()
            sample = default_collate([dataset[0]])
            x = sample[0]

        else: # run on DA-updated initial conditions
            if not use_huge:
                ic_path = download_input(date, ens)
            else:
                ic_path = f"{IC_INPUT_SOURCE_PATH}/{get_ic_relative_path(get_date_str(date), ens)}"
            x = process_input(ic_path, date, model.config.mesh)
            # use reencoder in forward to use DA updated ICs
            todo_dict = model.simple_gen_todo(todo, model.config.processor_dt)
            for dt in todo_dict:
                todo_str = todo_dict[dt]
                todo_dict[dt] = todo_str.replace('E,', 'rE,')
            todo = todo_dict

        
        out = {}
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            x = [xx.to('cuda') for xx in x]
            print(f"Generating forecasts for timesteps: {todo}", flush=True)
            ys = model.forward(x, todo, send_to_cpu=True, callback=save_hour)

    dispatch_ensemble_viz(ens)

    # TODO: ensmean doesn't work
    # if tots is not None:
    #     for h in out.keys():
    #         if h in tots:
    #             tots[h] += out[h].astype(np.float32)
    #         else:
    #             tots[h] = out[h].astype(np.float32)

    return True


def run_ensemble(date: datetime, use_da: bool, use_huge: bool, num_threads, idempotent=True):
    succeeded = 0
    if num_threads == 1:
        for member_num in RUN_MEMBERS:
            succeeded += run_member(date, member_num, use_da, use_huge, idempotent)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for member_num in RUN_MEMBERS:
                futures.append(executor.submit(run_member, date, member_num, use_da, use_huge, idempotent))

            for future in as_completed(futures):
                try:
                    succeeded += future.result()
                except Exception as e:
                    assert False, f"An error occurred: {str(e)}"
    if succeeded < len(RUN_MEMBERS):
        assert False, f"Only {succeeded} members succeeded, expected {len(RUN_MEMBERS)}"
    else:
        print(f"I solemnly swear all {len(RUN_MEMBERS)} members finished without error", flush=True)
    print("No mean calculation happening, still needs to be fixed", flush=True)
    
    # for h in tots.keys():
    #     gc.collect()
    #     start_time = time.time()
    #     print(f"Processing forecast hour {h}", flush=True)
    #     mean = tots[h] / float(len(DA_MEMBERS))
    #     print(f"Processed fh {h} ({round((time.time() - start_time) * 1000)} ms)", flush=True)
    #     start_time = time.time()
    #     save_upload_output(mean, h, date, 'mean')
    
def upload_worker():
    while True:
        task = upload_queue.get()
        if task is None:
            break
        output_fn = task
        upload_output(output_fn)
        upload_queue.task_done()

def save_upload_output(out, h, date, member):
    output_fn = save_output(out, h, date, member)
    if S3_OUTPUT_PATH is not None:
        upload_queue.put(output_fn)

def save_output(out, h, date, member):
    start_time = time.time()
    # Save output filename (including tags added by save_instance)
    output_fn = save_instance(out, f'{OUTPUT_PATH}/{get_date_str(date)}/ens{member}/{h}', model.config.output_mesh, DEFAULT_MODEL_NAME, upload_metadata=(S3_OUTPUT_PATH is not None))[len(OUTPUT_PATH)+1:]
    print(f"Saved fh {h} ({round((time.time() - start_time) * 1000)} ms)", flush=True)
    gc.collect()
    return output_fn

def upload_output(output_fn):
    print(f"Uploading {output_fn} to s3", flush=True)
    start_time = time.time()
    s3_fn = f'{S3_OUTPUT_PATH}/{output_fn}'
    S3_CLIENT.upload_file(f'{OUTPUT_PATH}/{output_fn}', BUCKET, s3_fn)
    print(f"Uploaded to s3 bucket {BUCKET} at {s3_fn} ({round((time.time() - start_time) * 1000)} ms)", flush=True)

def download_input(date: datetime, ens):
    date_str = get_date_str(date)
    s3_path = S3_INPUT_PATH + "/" + get_ic_relative_path(date_str, ens)
    local_path = IC_INPUT_PATH + "/" + get_ic_relative_path(date_str, ens)
    print(f"Downloading ICs for {date_str} and ens{ens} to {local_path}", flush=True)
    start_time = time.time()
    pathlib.Path(os.path.dirname(local_path)).mkdir(parents=True, exist_ok=True)
    S3_CLIENT.download_file(BUCKET, s3_path, local_path)
    print(f"Downloaded ICs for {date_str} and ens{ens} ({round((time.time() - start_time) * 1000)} ms)", flush=True)
    return local_path

def cleanup_input_dir(input_dir):
    print(f"Cleaning up {input_dir}", flush=True)
    start_time = time.time()
    for file in os.listdir(input_dir):
        os.remove(os.path.join(input_dir, file))
    os.removedirs(input_dir)
    print(f"Cleaned up {input_dir} ({round((time.time() - start_time) * 1000)} ms)", flush=True)

def cleanup_output_dir(date_str):
    output_dir = f"{OUTPUT_PATH}/{date_str}"
    print(f"Cleaning up {output_dir}", flush=True)
    start_time = time.time()
    # remove all subdirectories that start with 'ens'
    if not os.path.isdir(output_dir):
        print(f'{output_dir} doesnt exist, and not existing is even better than being clean', flush=True)
        return
    for file in os.listdir(output_dir):
        if file[:3] == 'ens' and os.path.isdir(os.path.join(output_dir, file)):
            shutil.rmtree(os.path.join(output_dir, file))
    os.removedirs(output_dir)
    print(f"Cleaned up {output_dir} ({round((time.time() - start_time) * 1000)} ms)", flush=True)

def run_if_needed(forecast_zero=None, use_da=False, use_huge=False, clean_all=False, num_threads=NUM_ENSEMBLE_THREADS, idempotent=True, output_path=None, no_s3=False):
    if forecast_zero is None:
        forecast_zero = latest_ens_date()
        print(f"Latest data: {forecast_zero} (latest GFS: {latest_gfs_date()})", flush=True)
    if isinstance(forecast_zero, str):
        forecast_zero = datetime.strptime(forecast_zero, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    
    kill_prev_process()
    load_model()
    global OUTPUT_PATH, S3_OUTPUT_PATH
    if not use_da:
        OUTPUT_PATH = '/fast/realtime/outputs/WeatherMesh'
        S3_OUTPUT_PATH = 'WeatherMesh'
    else:
        OUTPUT_PATH = '/huge/deep/realtime/outputs/WeatherMeshDA-out'
        #S3_OUTPUT_PATH = 'WeatherMeshDA-out'
    if output_path is not None:
        OUTPUT_PATH = output_path
    if no_s3:
        S3_OUTPUT_PATH = None

    if S3_OUTPUT_PATH is not None:
        upload_threads = []
        for _ in range(NUM_UPLOAD_THREADS):
            t = threading.Thread(target=upload_worker)
            t.daemon = True # so that upload workers don't block shutdown due to error
            t.start()
            upload_threads.append(t)

    run_ensemble(forecast_zero, use_da, use_huge, num_threads, idempotent)
    
    if S3_OUTPUT_PATH is not None:
        for _ in range(NUM_UPLOAD_THREADS):
            upload_queue.put(None)
        for t in upload_threads:
            t.join()

    # remove the downloaded ICs
    if not use_huge and use_da:
        cleanup_input_dir(IC_INPUT_PATH + "/" + get_date_str(forecast_zero))
    if clean_all:
        if use_da:
            assert False, "nah we want to keep DA outputs on /huge, not s3"
        cleanup_output_dir(get_date_str(forecast_zero))

if __name__ == '__main__':
    print(f"Started process {os.getpid()} for run_rt_ens", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('forecast_zero', nargs='?', help='Forecast zero time in YYYYMMDDHH format')
    parser.add_argument('--no-viz', action='store_true')
    parser.add_argument('--use_da', action='store_true', default=False)
    parser.add_argument('--use_huge', action='store_true', default=False)
    parser.add_argument('--clean-all', action='store_true', default=False) # clean output ens directories after uploading
    parser.add_argument('--num-threads', type=int, default=NUM_ENSEMBLE_THREADS)
    parser.add_argument('--not-idempotent', action='store_true', default=False)
    parser.add_argument('--output-path', type=str, default=None) # only to override default
    parser.add_argument('--no-s3', action='store_true', default=False)
    args = parser.parse_args()
    print(f"Running with args: {args}", flush=True)

    run_if_needed(forecast_zero=args.forecast_zero, use_da=args.use_da, use_huge=args.use_huge, clean_all=args.clean_all, num_threads=args.num_threads, idempotent=not args.not_idempotent, output_path=args.output_path, no_s3=args.no_s3)
