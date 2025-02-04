import torch
from collections import defaultdict
import gc

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    #torch.multiprocessing.freeze_support()
    torch.set_grad_enabled(False)

import pickle
import sys
sys.path.append('.')
sys.path.append("/fast/wbhaoxing/deep")
from utils import *
from eval import unnorm_output, unnorm, unnorm_output_partial, compute_errors, all_metric_fns
from data import *
from model_latlon_3d import *
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader    
import json
import hashlib
import base64
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
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a DataConfig attribute"
            setattr(self,k,v)
        self.update()

    def update(self):
        pass

def EvalModel(model_fn, dates, config : EvalConfig = None):
  with torch.no_grad():
    #eval_path = '/fast/evaluation/Neoreal_335M/'
    model = model_fn().to('cuda')

    eval_path = modelevalpath(model,model_fn.modelname)
    os.makedirs(f'{eval_path}/outputs/',exist_ok=True)
    os.makedirs(f'{eval_path}/errors/',exist_ok=True)
    inputs = model.config.inputs 
    #outputs = model.models[1].config.outputs ### HACKY: this is temp to just make this work for now
    outputs = model.config.outputs ### HACKY: this is temp to just make this work for now
    #outputs = model.config.outputs
    if dates is None:
        dates = get_dates((D(2020, 1, 1),D(2020,12,31)))
    print("date range", dates[0], dates[-1])
    print([m.source for m in inputs+outputs])
    if USE_ROLLOUT:
        dlts = get_rollout_times(config.dt_dict, time_horizon = config.time_horizon, min_dt = config.min_dt)
        if config.dt_override is not None:
            dlts = config.dt_override
    else:
        dlts = [k for k in config.model_proc_spec.keys()]

    print("dlts", dlts)
    dataset = WeatherDataset(DataConfig(inputs=inputs, outputs=outputs,
                        timesteps=dlts,
                        requested_dates = dates,
                        only_at_z = config.only_at_z,
                        clamp_output = np.inf,
                        ))

    print("hey dataset", dataset)
    
    eval_loop(model, dataset, eval_path, config)
            

def eval_loop(model, data, eval_path, config):
    sofar = defaultdict(list)
    data.check_for_dates()
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=5, collate_fn=default_collate)
    mesh = data.config.outputs[0]
    model = model.half()
    model.checkpointfn = None
    print("len data", len(data))
    #for i,sample in enumerate(dataloader):
    for i in range(len(data)):
        AAA = []
        #if os.path.exists(eval_out) and not rerun_files:
            #print(f'{eval_out} exists, skipping')
        #    pass#continue
        sample = default_collate([data[i]])
        #assert sample_dts(sample)[0] == dt, f"{dt} sample dt mismatch" ; assert int(sample[0][-1]) == nix, f"{nix} sample date mismatch"
        #with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        x = sample[0]
        print("helloooooo", [int((aa[-1]-x[-1])/3600) for aa in sample])
        x = [xx.to('cuda') for xx in x]
        #print("uhhh checkpoint", model.checkpointfn, model.config.checkpoint_every)
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            if USE_ROLLOUT:
                ys = model.rollout(x, dts=config.dt_override)
            else:
                ys = model(x, config.model_proc_spec, send_to_cpu=True)
        for j,(ddt,y) in enumerate(sorted(ys.items())):
            #nix = to_unix(data.config.sample_dates[i])
            nix = data.sample_array[i,0]
            assert sample[j+1][-1] == nix+ddt*3600, f"{nix} {ddt} {sample[j+1][-1]} {j} sample date mismatch"
            y = y.to('cuda')
            xu = unnorm(x[0],data.config.inputs[0])
            y = unnorm(y,data.config.outputs[0])
            #xu,y = unnorm_output(x[0],y,model,dt,y_is_deltas=False)
            yt = unnorm(sample[j+1][0].to('cuda'),mesh)
            if i % config.save_every == 0:
                if j==0:
                    save_instance(xu,f'{eval_path}/outputs/{to_filename(nix,0)}',mesh,config.model_name)
                save_instance(y,f'{eval_path}/outputs/{to_filename(nix,ddt)}',mesh,config.model_name)
            out = {}
            out['input_time'] = get_date_str(nix)
            out['output_time'] = get_date_str(nix+ddt*3600)
            out['forecast_dt_hours'] = int(ddt)
            #out['model_steps'] = model.last_steps
            out['model_input_source'] = data.config.inputs[0].source
            cmpsrc = data.config.outputs[0].source
            out['comparison_source'] = cmpsrc
            tag = config.tag
            out['tag'] = tag
            tag = tag if tag == '' else '.'+tag
            eval_out = f'{eval_path}/errors/{get_date_str(nix)}+{ddt}.vs_{cmpsrc}{tag}.json'
            with torch.autocast(enabled=False, device_type='cuda', dtype=torch.float32):
                for error_name,metric_fn in all_metric_fns.items():
                    #if error_name != "rmse": continue
                    metrics = compute_errors(y,yt,mesh,metric_fn=metric_fn)
                    out[error_name] = metrics
            with open(eval_out,'w') as f:
                json.dump(out,f,indent=2)
            sofar[ddt].append(out["rmse"]["129_z_500"])
            print(f'{get_date_str(nix)}, dt:{ddt}, {i}/{len(dataloader)}, z500 {out["rmse"]["129_z_500"]}')
            for d in sorted(sofar.keys()):
                print("sofar", d, "rms", np.sqrt(np.mean(np.square(sofar[d]))))
            AAA.append((ddt,out["rmse"]["129_z_500"]))
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
EVAL_2020_DATES_MAR = get_dates((D(2020, 1, 1),D(2020,3,31)))
#EVAL_2020_DATES_LITE = get_dates((D(2020, 1, 1),D(2020,12,31), timedelta(days=3)))
#EVAL_2020_DATES = get_dates((D(2020, 10, 1),D(2020,12,31)))

if len(sys.argv) > 1:
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    EVAL_2020_DATES = np.array_split(EVAL_2020_DATES, b)
    EVAL_2020_DATES = EVAL_2020_DATES[a]
    #print("initially", len(ADAPTER_EVAL_DATES2), [len(x) for x in ADAPTER_EVAL_DATES2])
    #ADAPTER_EVAL_DATES2 = ADAPTER_EVAL_DATES2[a]
    print("Did a subset", len(EVAL_2020_DATES))

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

    modelfn = get_rtyamahabachelor5
    modelfn = get_rtyblong
    #modelfn = get_gen1_hegelquad
    modelfn = get_naives2s #59M params

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
        proc_spec = ForecastStep3D.simple_gen_todo(
            list(range(0,25,1)) + list(range(36,145,12)) + list(range(168,24*14+1,24)),
            [1,6]
        )

    #EvalModel(modelfn, ADAPTER_EVAL_DATES2, EvalConfig(only_at_z=[0], model_proc_spec = proc_spec, model_name = modelfn.__name__[4:]))#,9,12,18,24]))
    
    proc_spec = ForecastStep3D.simple_gen_todo(list(range(0,25,1)) + list(range(36,145,12)), [1,6])
    #proc_spec = ForecastStep3D.simple_gen_todo(list(range(0,145,12)) + list(range(168,220,24)), [1,6])
    for x in proc_spec:
        proc_spec[x] = proc_spec[x].replace('P6','P3,P3')
    from pprint import pprint
    proc_spec = ForecastStep3D.simple_gen_todo(list(range(24,217,24)), [1,3])
    pprint(proc_spec)

    modelfn = get_ducatidoctorate
    modelfn = get_latentdoctorate
    #EvalModel(modelfn, ADAPTER_EVAL_DATES3, EvalConfig(only_at_z=[0], model_proc_spec = proc_spec, model_name = modelfn.__name__[4:]))#,9,12,18,24]))
    #EvalModel(modelfn, EVAL_2020_DATES_LITE, EvalConfig(only_at_z=[0,6,12,8], model_proc_spec = proc_spec, model_name = modelfn.__name__[4:]))#,9,12,18,24]))
    EvalModel(modelfn, EVAL_2020_DATES, EvalConfig(only_at_z=[0,12], model_proc_spec = proc_spec, model_name = "test"+modelfn.__name__[4:]))#,9,12,18,24]))
    # EvalModel(get_rtyamahabachelor, ADAPTER_EVAL_DATES2, EvalConfig(only_at_z=[0,12], model_proc_spec = proc_spec))#,9,12,18,24]))
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




