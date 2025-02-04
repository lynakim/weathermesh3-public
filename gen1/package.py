import os
import shutil
import torch
import torch.nn as nn

from gen1.data import NeoWeatherDataset, NeoDataConfig
from gen1.meshes import LatLonGrid
from gen1.utils import get_dates, D, levels_gfs, levels_medium, levels_tiny, levels_joank
from gen1.model import ForecastStepSwin3D, ForecastStepConfig, ForecastStepCombo


EVALUATION_PATH = '/huge/deep/evaluation/'

def modelfullname(model,name):
    params = sum(p.numel() for p in model.parameters())
    return f'{name}_{int(params/1e6)}M'

def modelevalpath(model,name):
    return f'{EVALUATION_PATH}/{modelfullname(model,name)}'

def weightspath(model,load_path,name):
    return f'{modelevalpath(model,name)}/weights/{os.path.basename(load_path)}'

def package_(model,load_path,name):
    print("New package, no pickles")
    checkpoint = torch.load(load_path,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    save_path = weightspath(model,load_path,name)
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    shutil.copy(load_path, save_path)
    print("Saved to", save_path)
    return model

def package(load_path):
    def inner(model_func):
        assert model_func.__name__.startswith('get_'), "model_func must be named get_something"
        name = model_func.__name__[4:]
        def wrapper():
            model = model_func()
            if load_path is None:
                return model
            weights = weightspath(model,load_path,name)
            print("loading", weights)
            if not os.path.exists(weights):
                return package_(model,load_path,name)
            # gen1 assumes the checkpoint to be state dict only
            sd = torch.load(weights, map_location='cpu')
            model.load_state_dict(sd,strict=False)
            model.name = name
            return model 
        wrapper.__name__ = model_func.__name__
        wrapper.modelname = name
        
        return wrapper
    return inner

@package('/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/state_dict_epoch5_iter547188_loss0.088.pt')
def get_neoquadripede():
    mesh = LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[6,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], dims_per_head=32, processor_dt=6, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=False))
    model.output_deltas = True
    model.do_sub = False
    return model

@package(None)
def get_hegelcasioquad():
    model1 = get_neoquadripede()
    model2 = get_neocasio()
    adapter = get_neohegel()
    model = ForecastStepCombo([model2,model1],adapter=adapter)
    return model


@package(None)
def get_hegelcasiopony():
    model1 = get_shallowpony()
    model2 = get_neocasio()
    adapter = get_neohegel()
    model = ForecastStepCombo([model1, model2],adapter=adapter)
    return model

@package('/fast/ignored/runs/run_Jan16-neocasio_20240116-183046/state_dict_epoch2_iter271192_loss0.051.pt')
def get_neocasio():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_model_epoch93_iter89982_loss0.066.ptvars=extra)
    input = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    output = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    timesteps = [1, 3, 6]
    model = ForecastStepSwin3D(ForecastStepConfig([input], outputs=[output], patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=False, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64, neorad=True))
    return model

@package("/fast/ignored/runs_adapt/run_Feb1-legeh-911-reprise_20240201-133247/state_dict_epoch56_iter54392_loss0.057.pt")
def get_neohegel():
    mesh1 = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25', input_levels=levels_gfs, levels=levels_medium)
    #mesh1 = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    mesh2 = LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    import gen1.model_unet as model_unet
    model_conf = ForecastStepConfig([mesh1, mesh2],
                outputs = [mesh3],
                patch_size=(4,8,8), 
                hidden_dim=768, 
                enc_swin_depth=0,
                dec_swin_depth=0, 
                proc_swin_depth=0, 
                adapter_swin_depth=8,
                timesteps=[0], 
                output_deltas = True,
                adapter_use_input_bias=True,
                use_matepoint = False,
                processor_dt = -1,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True),
                )

    adapter_model = model_unet.ForecastStepAdapterConv(model_conf)
    return adapter_model

@package('/fast/ignored/runs/run_Feb18-shallowpony_20240218-184013/state_dict_epoch4_iter479992_loss0.088.pt')
def get_shallowpony():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_model_epoch93_iter89982_loss0.066.ptvars=extra)
    input = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    output = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    output = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    input = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    output = LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24, 72]
    timesteps = [24]
    #model = ForecastStepSwin3D(ForecastStepConfig([input], outputs=[output], patch_size=(4,8,8), hidden_dim=928, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=10, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=False, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(4,3,6)))
    model = ForecastStepSwin3D(ForecastStepConfig([input], outputs=[output], patch_size=(5,8,8), hidden_dim=1504, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=4, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=False, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=128, neorad=True, window_size=(2,3,6)))
    return model

@package(None)
def get_hegelquad():
    model = get_neoquadripede()
    adapter = get_neohegel()
    model = ForecastStepCombo([model],adapter=adapter)
    return model