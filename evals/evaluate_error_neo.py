import torch
from collections import defaultdict
import gc

from datasets import AnalysisDataset, DailyAnalysisDataset

from train.trainer import collate_fn

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    #torch.multiprocessing.freeze_support()
    torch.set_grad_enabled(False)

import pickle
import sys
sys.path.append('.')
sys.path.append("/fast/wbhaoxing/deep")
from utils import *
from eval import unnorm, compute_errors, all_metric_fns
from data import *
from model_latlon_3d import *
import json
from evals.package_neo import *
from evals import *
from evaluate_error import *
from gen1.package import get_hegelquad as get_gen1_hegelquad

USE_ROLLOUT = False

class EvalConfig():
    def __init__(self, **kwargs):
        self.tag = ''
        self.time_horizon = 24
        self.min_dt = 1
        self.only_at_z = [0,12]
        self.save_every = 50
        self.dt_dict = {}
        self.dt_override = None
        self.model_proc_spec = {}
        self.model_name = None
        self.rerun_files = False
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a DataConfig attribute"
            setattr(self,k,v)
        self.update()

    def update(self):
        pass

def EvalModel(model_fn, dates, input_datasets, output_datasets, config : EvalConfig = None, eval_every = 1, s2s=False):
  # setting eval_every = 3 evaluates every 3rd date requested from dataloader eg. to evenly sample from 2020
  with torch.no_grad():
    #eval_path = '/fast/evaluation/Neoreal_335M/'
    model = model_fn().to('cuda')
    model.config.checkpoint_type = "none"
    eval_path = modelevalpath(model,model_fn.modelname)
    if dates is None:
        dates = get_dates((D(2020, 1, 1),D(2020,12,31)))
    print("date range", dates[0], dates[-1])
    if USE_ROLLOUT:
        dlts = get_rollout_times(config.dt_dict, time_horizon = config.time_horizon, min_dt = config.min_dt)
        if config.dt_override is not None:
            dlts = config.dt_override
    else:
        dlts = [k for k in config.model_proc_spec.keys()]

    input_meshes = model.config.inputs 
    output_meshes = model.config.outputs
    print([m.source for m in input_meshes+output_meshes])
    input_datasets = [d(input_meshes[i]) for i, d in enumerate(input_datasets)]
    output_datasets = [d(output_meshes[i]) for i, d in enumerate(output_datasets)]
    print("dlts", dlts)
    dataset = WeatherDataset(DataConfig(inputs=input_datasets, outputs=output_datasets,
                        timesteps=dlts,
                        requested_dates = dates,
                        only_at_z = config.only_at_z,
                        clamp_output = np.inf,
                        ))
    
    eval_loop(model, dataset, eval_path, config, eval_every, s2s=s2s)
            

def eval_loop(model, data, eval_path, config, eval_every, s2s=False):
    sofar = defaultdict(list)
    data.check_for_dates()
    model = model.half()
    model.checkpointfn = None
    config.checkpoint_type = "none"
    print("len data", len(data))
    assert len(model.decoders) == 1, f"only supports 1 decoder rn"
    output_mesh = data.config.outputs[0].mesh
    cmpsrc = output_mesh.source
    tag = config.tag
    weights = model.decoders[0].geospatial_loss_weight
    if s2s:
        resolution = 1.5
        weights = weights[:,::6,::6,:]
    else:
        resolution = 0.25
    max_dt = max(config.model_proc_spec.keys())
    n_dates = len(data)
    os.makedirs(f'{eval_path}/outputs/',exist_ok=True)
    error_dir = f'{eval_path}/errors{"_1.5deg" if s2s else ""}/'
    os.makedirs(error_dir,exist_ok=True)
    
    for i in range(n_dates):
        if i % eval_every != 0: continue
        nix = data.sample_array[i,0]
        if not config.rerun_files:
            last_error_file = f'{eval_path}/errors/{get_date_str(nix)}+{max_dt}.vs_{cmpsrc}{tag}.json'
            if os.path.exists(last_error_file):
                print(f'{last_error_file} exists, skipping {get_date_str(nix)}')
                continue
        sample = collate_fn([data[i]])
        #assert sample_dts(sample)[0] == dt, f"{dt} sample dt mismatch" ; assert int(sample[0][-1]) == nix, f"{nix} sample date mismatch"
        #with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        assert len(sample.inputs) == 1, f"only handles 1 input rn"
        #x = sample.inputs[0][1]
        ts = sample.timestamps
        #x_t0 = [sample.inputs[0][1][0], torch.tensor([ts[0]])]
        print("helloooooo", [int((aa-ts[0])/3600) for aa in ts])
        x = [xx.to('cuda') for xx in sample.get_x_t0()]
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            if USE_ROLLOUT:
                ys = model.rollout(x, dts=config.dt_override)
            else:
                ys = model(x, config.model_proc_spec)#, callback=calc_and_save_errors) #, send_to_cpu=True)
        ys = sorted([y for y in ys.items() if y[0]!='latent_l2'])
        for j,(ddt,y) in enumerate(ys):
            #nix = to_unix(data.config.sample_dates[i])
            assert ts[j+1] == nix+ddt*3600, f"{nix} {ddt} {ts[j+1]} {j} sample date mismatch"
            if type(y) == list:
                y = y[0]
            y = y.to('cuda')
            #xu = unnorm(x[0],data.config.inputs[0])
            y = unnorm(y,output_mesh, exp_log_vars=True)
            yt = unnorm(sample.outputs[j][1][0].to('cuda'),output_mesh, exp_log_vars=True)
            if config.save_every is not None and i % config.save_every == 0:
                save_instance(y,f'{eval_path}/outputs/{to_filename(nix,ddt)}',output_mesh,config.model_name)
            out = {}
            out['input_time'] = get_date_str(nix)
            out['output_time'] = get_date_str(nix+ddt*3600)
            out['forecast_dt_hours'] = int(ddt)
            #out['model_steps'] = model.last_steps
            out['model_input_source'] = data.config.inputs[0].source
            out['comparison_source'] = cmpsrc
            out['tag'] = tag
            out['grid_resolution'] = resolution
            tag = tag if tag == '' else '.'+tag
            eval_out = f'{error_dir}/{get_date_str(nix)}+{ddt}.vs_{cmpsrc}{tag}.json'
            with torch.autocast(enabled=False, device_type='cuda', dtype=torch.float32):
                for error_name,metric_fn in all_metric_fns.items():
                    if error_name not in ["rmse", "bias"]: continue
                    metrics = compute_errors(y,yt,weights, output_mesh, metric_fn=metric_fn, only_sfc=s2s, resolution=resolution)
                    out[error_name] = metrics
            with open(eval_out,'w') as f:
                json.dump(out,f,indent=2)
            ref_var = "129_z_500" if not s2s else "167_2t"
            print(f'{get_date_str(nix)}, dt:{ddt}, {i}/{n_dates}, {ref_var} {out["rmse"][ref_var]}')
            sofar[ddt].append(out["rmse"][ref_var])
        for d in sorted(sofar.keys()):
            print("sofar", d, "rms", np.sqrt(np.mean(np.square(sofar[d]))))
            #AAA.append((ddt,out["rmse"]["129_z_500"]))
        continue
        plt.plot(*zip(*AAA))
        plt.grid()
        plt.savefig('ignored/ohp.png')
        gc.collect()
            


def germany_error(y,yt,mesh):
    bbox = (47,5,55,15)
    return compute_errors(y,yt,mesh,bbox=bbox)

ADAPTER_EVAL_DATES = get_dates((D(2022, 1, 1), D(2022, 3, 1)))
ADAPTER_EVAL_DATES2 = get_dates((D(2022, 8, 1),D(2022,12,30)))
ADAPTER_EVAL_DATES3 = get_dates((D(2024, 3, 10),D(2024, 5, 15)))
EVAL_2020_DATES = get_dates((D(2020, 1, 1),D(2020,12,31)))
#EVAL_2020_DATES = get_dates((D(2020, 1, 1),D(2020,2,20)))
requested_dates = EVAL_2020_DATES

fill_date_gaps = False
if len(sys.argv) > 1:
    fill_date_gaps = True # when we break up dates into chunks, there are extra gaps = forecast days
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    requested_dates = np.array_split(requested_dates, b)
    print("initially", len(requested_dates), [len(x) for x in requested_dates])
    requested_dates = requested_dates[a]
    print("Did a subset", len(requested_dates))
    is_last = a == (b-1)

if __name__ == '__main__':
    #EvalModel(get_neoquadripede, EVAL_2020_DATES, error_fn=germany_error, tag='germany')
    #EvalModel(get_shortking, EVAL_2020_DATES, timesteps=[48])
    #EvalModel(get_handfist, EVAL_2020_DATES, timesteps=[24])
    #EvalModel(get_shortking, EVAL_2020_DATES, timesteps=[24, 48, 72, 96, 120])
    #EvalModel(get_widepony, EVAL_2020_DATES, timesteps=[24, 48, 72, 96, 120, 144])
    #EvalModel(get_widepony, EVAL_2020_DATES, EvalConfig(time_horizon=3*24, only_at_z=[0,12], min_dt=24))
    #EvalModel(get_shallowpony, EVAL_2020_DATES, EvalConfig(time_horizon=3*24, only_at_z=[0,12], min_dt=24))
    #EvalModel(get_wrappy, EVAL_2020_DATES, EvalConfig(time_horizon=5*24, only_at_z=[0,12], min_dt=24))
    #EvalModel(get_ultralatentbachelor, EVAL_2020_DATES, EvalConfig(time_horizon=9*24, only_at_z=[0,12], min_dt=24))
    #EvalModel(get_highschooler, EVAL_2020_DATES, EvalConfig(time_horizon=5*24, only_at_z=[0,12], min_dt=24))
    #EvalModel(get_longmaster, EVAL_2020_DATES, EvalConfig(time_horizon=5*24, only_at_z=[0,12], min_dt=24))
    #EvalModel(get_master, EVAL_2020_DATES, EvalConfig(time_horizon=3*24, only_at_z=[0,12], min_dt=24))

    #EvalModel(get_shallowpony, EVAL_2020_DATES, timesteps=[24, 48, 72])
    #EvalModel(get_rpshortking, EVAL_2020_DATES, timesteps=[24, 48, 72, 96, 120])
    #EvalModel(get_shortquad, EVAL_2020_DATES, timesteps=[24, 48, 72, 96, 120])
    ###EvalModel(get_shortking, EVAL_2020_DATES, timesteps=[24, 48, 72, 96, 120])
    #EvalModel(get_shortquad, EVAL_2020_DATES,EvalConfig(time_horizon=24*5,min_dt=24))
    #for step in range(1,31):
    #EvalModel(get_neocasioquad, EVAL_2020_DATES,timesteps=[48])
    #EvalModel(get_hegelcasioquad, ADAPTER_EVAL_DATES2,timesteps=[24])
    #EvalModel(get_hegelcasio, ADAPTER_EVAL_DATES2,timesteps=[24])

    #EvalModel(get_stupidyolo, EVAL_2020_DATES, EvalConfig(only_at_z=[0,12], dt_override=[24,24*6,24*12,24*24]))
    #EvalModel(get_gfshresbachelor, ADAPTER_EVAL_DATES2, EvalConfig(only_at_z=[0,12], dt_override=[6,12,18,24,36,48,60,72,84,96,108,120,132,144]))

    def full_latent(total_dt,proc_dt):
        assert total_dt % proc_dt == 0
        return 'E,' + f"P{proc_dt},"*(total_dt // proc_dt) + 'D'

    #modelfn = get_rtyamahabachelor5
    #modelfn = get_rtyblong
    #modelfn = get_gen1_hegelquad
    modelfn = get_fenrir
    # default args
    is_s2s = False
    input_datasets = [AnalysisDataset]
    output_datasets = [AnalysisDataset]
    

    if modelfn.__name__ == 'get_rtyamahabachelor':
        proc_spec = {
            0: 'E,D',                                    #25.7
            #0: 'E,D,rE,D',                              #246
            #3: 'E,P1,P1,P1,D',                          #54.5
            3: 'E,D,rE,P1,P1,P1,D',                      #119.6
            #6:'E,D,rE,P1,P1,P1,P1,P1,P1,D',
            #6: 'E,P6,D',                                # 25.600616919187647
            6: 'E,D,rE,P6,D',                            # 32.14548204210491     
            #9: 'E,P6,D,rE,P1,P1,P1,D',                  # 26.781219615012592
            #12: 'E,P6,P6,D',                            # 40.71994590566578
            12: 'E,P6,D,rE,P6,D',                        # 34. 

            #15: 'E,P6,P6,D,rE,P1,P1,P1,D',                 #50.3215289086380 
            #15: 'E,P6,D,rE,P6,D,rE,P1,P1,P1,D',            #38.333693095071695
            #18: 'E,P6,P6,P6,D',                            #66.08714024592847
            #18: 'E,P6,D,rE,P6,D,rE,P6,D',                  #36.87909713054849          
            #24: 'E,P6,P6,P6,P6,D',                         #39.47375578325579
            #27: 'E,P6,P6,P6,P6,P1,P1,P1,D',                #135.63622800533776
            #27: 'E,P6,P6,P6,P6,D,rE,P1,P1,P1,D',           #47.71551147668697
            #30: 'E,P6,P6,P6,P6,P6,D',                      #66.66887650704514
            30: 'E,P6,P6,P6,P6,D,rE,P6,D',                  #43.36782493132109          

            #48: 'E,P6,P6,P6,P6,P6,P6,P6,P6,D',             #67.33970416827891

        }
    if modelfn.__name__ == 'get_yamahabachelor':
        proc_spec = ForecastStep3D.simple_gen_todo(range(0,50,3),[1,6])
    if modelfn.__name__ in ['get_rtyamahabachelor2', 'get_rtyamahabachelor5']:
        proc_spec = ForecastStep3D.simple_gen_todo(
            #list(range(0,25,1)) + list(range(36,145,12)) + list(range(168,220,24)),
            [6,24],
            [1,6]
        )
    # if modelfn.__name__ in ['get_rtyamahabachelor2', 'get_rtyamahabachelor5']:
    #     proc_spec = ForecastStep3D.simple_gen_todo(list(range(0,25,1)), [1,6])
    if modelfn.__name__ == 'get_rtyamahabachelor5':
        proc_spec = ForecastStep3D.simple_gen_todo(list(range(24, 24*14+1, 24)), [1,6])
    if modelfn.__name__ == 'get_rtyblong':
        # proc_spec = ForecastStep3D.simple_gen_todo(
        #     list(range(0,25,1)) + list(range(36,145,12)) + list(range(168,24*14+1,24)),
        #     [1,6]
        # )
        proc_spec = ForecastStep3D.simple_gen_todo(list(range(0,24*14+1,24)), [1,6])

    #EvalModel(modelfn, ADAPTER_EVAL_DATES2, EvalConfig(only_at_z=[0], model_proc_spec = proc_spec, model_name = modelfn.__name__[4:]))#,9,12,18,24]))
    
    if modelfn.__name__ in ['get_naives2s', 'get_naives2s_short', 'get_cas2sandra']:
        proc_spec = ForecastStep3D.simple_gen_todo(list(range(0,24*30+1,24)), [24])

    if modelfn.__name__ in ['get_fresh', 'get_stale']:
        proc_spec = ForecastStep3D.simple_gen_todo(list(range(0,24*5,6)) + list(range(24*5,24*14+1,24)), [6])

    if modelfn.__name__ in ['get_thor', 'get_wodan', 'get_freyja', 'get_tyr', 'get_fenrir']:
        #proc_spec = ForecastStep3D.simple_gen_todo(list(range(0,24*45+1,24)), [24])
        proc_spec = ForecastStep3D.simple_gen_todo(list(range(0,24*2+1,24)), [24])
        input_datasets = [DailyAnalysisDataset]
        output_datasets = [DailyAnalysisDataset]
        is_s2s = True
        #proc_spec = ForecastStep3D.simple_gen_todo(list(range(0,24*2+1,24)), [24])

    #EvalModel(modelfn, ADAPTER_EVAL_DATES3, EvalConfig(only_at_z=[0], model_proc_spec = proc_spec, model_name = modelfn.__name__[4:]))#,9,12,18,24]))
    #EvalModel(modelfn, EVAL_2020_DATES_LITE, EvalConfig(only_at_z=[0,6,12,8], model_proc_spec = proc_spec, model_name = modelfn.__name__[4:]))#,9,12,18,24]))
    if fill_date_gaps and not is_last:
        requested_dates = np.append(requested_dates, [requested_dates[-1] + timedelta(days=i) for i in range(1,max(proc_spec.keys())//24 +1)])
        print(f'Extended requested dates by {max(proc_spec.keys())//24 +1} days, now {requested_dates[0].strftime("%Y-%m-%d")} to {requested_dates[-1].strftime("%Y-%m-%d")}')

    EvalModel(modelfn, requested_dates, input_datasets, output_datasets, EvalConfig(only_at_z=[0], model_proc_spec = proc_spec, model_name = modelfn.__name__[4:], save_every=None), s2s=is_s2s) #, eval_every=3)
    # EvalModel(get_rtyamahabachelor, ADAPTER_EVAL_DATES2, EvalConfig(only_at_z=[0,12], model_proc_spec = proc_spec))#,9,12,18,24))
    #EvalModel(get_yamaha, ADAPTER_EVAL_DATES2, EvalConfig(only_at_z=[0,12], dt_override=[3,6]))
    #EvalModel(get_bachelor, ADAPTER_EVAL_DATES2, EvalConfig(only_at_z=[0,12], dt_override=[6,12,18,24]))
    #from gen1.package import get_neoquadripede
    #EvalModel(get_neoquadripede, ADAPTER_EVAL_DATES2, EvalConfig(only_at_z=[0,12], dt_override=[6,12,18,24]))
    # EvalModel(get_gen1_hegelquad, ADAPTER_EVAL_DATES2, EvalConfig(only_at_z=[0,12], dt_override=list(range(6,25,6)) + list(range(36,145,12)) + list(range(168,220,24))))



    if 0:
        EvalModel(get_hegelcasioquad, get_dates((D(2022, 9, 25),D(2022, 10, 20))),
            EvalConfig( 
                time_horizon=24, 
                min_dt=1, 
                dt_dict={24:6},
                only_at_z = [0],
                save_every = 1
            )
        )

    if 0:
        EvalModel(get_ultralatentbachelor, get_dates((D(2022, 9, 21),D(2022, 10,4))),
            EvalConfig( 
                dt_override=[24,48,72,96,120],
                only_at_z = [0],
                save_every = 1
            )
        )

    #EvalModel(get_neoquadripede, EVAL_2020_DATES,timesteps=[24]
    #EvalModel(get_neoquadripede, EVAL_2020_DATES,timesteps=[i*12 for i in range(1,14)])

# The below command is evaluating the 4th of 6 chunks 
# CUDA_VISIBLE_DEVICES=3 python3 evals/evaluate_error_neo.py 3 6 