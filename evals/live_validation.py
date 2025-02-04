from utils import *
from torch.utils.tensorboard import SummaryWriter
from eval import unnorm_output, unnorm, unnorm_output_partial, compute_errors, all_metric_fns
from evals.package_neo import load_weights
from data import WeatherDataset, DataConfig
from model_latlon_3d import *
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader    
from evals.package_neo import *
from evals import *
import subprocess



NUM_SAMPLES = 4

def run_validation(model,weights):
    log_dir = os.path.dirname(weights)
    step = int(os.path.basename(weights).split("step")[1].split("_")[0])
    load_weights(model,weights)
    dates = get_dates((D(2020, 1, 1),D(2020,12,31)))
    inputs = model.config.inputs 
    outputs = model.config.outputs
    dts = model.config.timesteps
    data = WeatherDataset(DataConfig(inputs=inputs, outputs=outputs,
                    timesteps=dts,
                    requested_dates = dates,
                    use_mmap = True,
                    clamp_output = np.inf,
                    ))
    data.check_for_dates()
    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=2, collate_fn=default_collate)
    mesh = data.config.outputs[0]
    model.eval()
    model.cuda()
    model.half()
    model.config.checkpointfn = checkpoint.checkpoint

    all = {dt:[] for dt in dts}
    with torch.no_grad():
        for i,sample in enumerate(dataloader):
            if i >= NUM_SAMPLES:
                break
            t0 = int(sample[0][-1].item())
            print(get_date(t0))
            x = sample[0]
            x = [xx.cuda() for xx in x]
            for j,dt in enumerate(dts):
                with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
                    y = model(x,dt=dt)
                y = unnorm(y,mesh)
                yt = unnorm(sample[j+1][0].to('cuda'),mesh)
                metrics = compute_errors(y,yt,mesh,metric_fn=all_metric_fns['rmse'])
                all[dt].append(metrics)
    del dataloader

    for dt in dts:
        all[dt] = {k:np.sqrt(np.mean(np.square([m[k] for m in all[dt]]))) for k in all[dt][0].keys()}
 
    #print(all)
    writer = SummaryWriter(log_dir=log_dir,filename_suffix='validation')
    for dt in dts:
        print(f'Validation_{dt} z 500:',all[dt]['129_z_500'])
        for k,v in all[dt].items():
            if k.count("_") == 2 and not "_500" in k:
                continue
            #print(k)
            writer.add_scalar(f'Validation_{dt}/{k}',v,step)
    writer.close()
    time.sleep(1) # this sleep is to avoid any file flushing weirdness that might happen if we try to combine the files too quickly
    #combine_validation_results(log_dir)

    print("Validation complete for ",weights)

def combine_validation_results(log_dir):
    from tensorboard.backend.event_processing import event_accumulator

    event_files = [
        os.path.join(log_dir, f) for f in os.listdir(log_dir)
        if f.endswith('validation') and 'tfevents' in f
    ]

    writer = SummaryWriter(log_dir=log_dir, filename_suffix='validation')

    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        for tag in ea.Tags()['scalars']:
            for event in ea.Scalars(tag):
                writer.add_scalar(tag, event.value, event.step)
        print("Removing",event_file)
        os.remove(event_file)
    writer.close()

#export CUDA_VISIBLE_DEVICES=3 ; python3 runs/Nov12.py Aug19_bachelorette validate /huge/deep/runs/run_Aug19-bachelorette_20240819-154349/model_epoch1_iter156632_step19579_loss0.072.pt

def get_most_recent_tfeventsfile(log_dir):
    files = [f for f in os.listdir(log_dir) if 'tfevents' in f and 'validation' not in f]
    if not files:
        return None
    most_recent_file = max(files, key=lambda f: int(f.split('.')[3]))
    return os.path.join(log_dir, most_recent_file)

def find_text_summary(event_file):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    for tag in ea.Tags()['tensors']:
        if "Cmd String" in tag:
            return ea.Tensors(tag)[0].tensor_proto.string_val[0].decode('utf-8')
        

def launch(weightsfile):
    rundir = os.path.dirname(weightsfile)
    event_file = get_most_recent_tfeventsfile(rundir)
    try:
        cmd = find_text_summary(event_file)
    except ValueError as e:
        print(RED(f"Could not find cmd string in {event_file}"))
        return
    run = cmd.split()[-1]
    file = "/".join(cmd.split()[0].split("/")[-2:])
    runloc = "/".join(cmd.split()[0].split("/")[:-2])
    runloc = "/fast/djohn"  ## TODO REMOVE THIS
    assert file.startswith('runs/'), "Ok I don't know how you were running this thing."

    #val_cmd = ["sleep", "10", ";", "cd",runloc,";","export","CUDA_VISIBLE_DEVICES=3",";","python3",file,run,'validate',weightsfile]


    # the sleep 10 is to give the wieghts file time to be written to disk. Hacky but simple and probably good enough.    
    val_cmd = f"sleep 10; cd {runloc}; export CUDA_VISIBLE_DEVICES=3; python3 {file} {run} validate {weightsfile}"
    print(RED(f"LAUNCHING: {val_cmd}"))
    subprocess.Popen(val_cmd, shell=True)


def get_all_files(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return set(file_list)





if __name__ == "__main__":
    from tensorboard.backend.event_processing import event_accumulator

    path = "/huge/deep/runs/"
    existing_files = get_all_files(path)

    #launch("/huge/deep/runs/run_Aug19-bachelorette_20240819-154349/model_epoch1_iter160312_step20039_loss0.066.pt")
    
    while True:
        time.sleep(1)
        current_files = get_all_files(path)
        new_files = current_files - existing_files

        for new_file in new_files:
            if new_file.endswith('.pt'):
                print(RED(f"New weights file found: {new_file}"))
                launch(new_file)
        
        existing_files = current_files


    
    

    
