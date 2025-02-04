from launch import *

if am_i_torchrun():
    from train import *

from train import * # <--- comment this back in if not using DDP
from torch.utils import checkpoint as torch_checkpoint

# Model Names to Use: Star Wars , House DJs
# Luke, Leia, Han, Vader, Fett
# Caribou, Jersey, Punk, Again, 

# Test launch just to step through the model code
@launch(ddp=False, kill_nvidia=False)
def Nov22_model_decoder_debug_testing():
    # Debug 
    config.nope = True
    
    tdates = get_dates([(D(1971, 1, 1), D(2022, 12, 31), timedelta(hours=3))])
    
    # Gather mesh variables using NeoDatasetConfig (lives in utils.py)
    extra = ['tc-maxws', 'tc-minp'] # Cyclone max wind speed, Cyclone min pressure
    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    mesh2 = meshes.LatLonGrid(source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    #tc_mesh = meshes.LatLonGrid()'
    
    timesteps = [6,24,72]
    data = WeatherDataset(DataConfig(inputs=[imesh, mesh2], outputs=[omesh, mesh2],
                                     timesteps=timesteps,
                                     requested_dates=tdates,
                                     only_at_z=list(range(0, 24, 3))))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=False, 
            neorad=True, 
            window_size=(3,5,7),
            parallel_encoders=True))

    config.prefetch_factor = 2
    config.ignore_train_safegaurd = True
    config.log_every = 10
    config.save_every = 100
    config.save_imgs_every = 1000
    config.HALF = True
    config.pin_memory = True
    
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    
    config.lr_sched.cosine_en = True
    config.lr_sched.div_factor = 4
    config.lr_sched.step_offset = 0
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.cosine_period = 40_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.lr = 1e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    
    config.timeout = timedelta(minutes=20)
    
    w = WeatherTrainer(conf=config, model=model, data=data)
    w.run()

# Input output but with altered baseline model
# With extra variables
@launch(nodes={'singing': 2}, port=29500, start_method="spawn", zulip=True, ping='@**Jack Michaels**')
def Nov22_TCfullforce_try2():
    config.resume = "_"+config.activity.replace("_","-")+"_"
    
    tdates = get_dates([(D(1971, 1, 1), D(2022, 12, 31), timedelta(hours=3))])
    
    # Gather mesh variables using NeoDatasetConfig (lives in utils.py)
    extra = ['tc-maxws', 'tc-minp'] # Cyclone max wind speed, Cyclone min pressure
    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    
    timesteps = [6,24,72]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                     timesteps=timesteps,
                                     requested_dates=tdates,
                                     only_at_z=list(range(0, 24, 3))))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=False, 
            neorad=True, 
            window_size=(3,5,7)))

    #config.num_workers = 2
    config.prefetch_factor = 2
    #config.compute_Bcrit_every = 10
    config.ignore_train_safegaurd = True
    config.log_every = 10
    config.save_every = 100
    config.save_imgs_every = 1000
    config.HALF = True
    config.pin_memory = True
    
    config.gpus = '3-4'
    
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    
    config.lr_sched.cosine_en = True
    config.lr_sched.div_factor = 4
    config.lr_sched.step_offset = 0
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.cosine_period = 40_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.lr = 1e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    
    config.timeout = timedelta(minutes=20)
    
    w = WeatherTrainer(conf=config, model=model, data=data)
    w.run()

# Runs baseline TC model with variable weights for extra output variables
# (levels_medium) for output levels
# Changed variable weights from 'variable_weights_28' to 'tc_variable_weights_28' in train.py line 208
@launch(nodes={'baga': 2}, port=29502, start_method="spawn", kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')
def Nov21_TCvariableweights_try2():
    # Resuming config
    config.resume = "_"+config.activity.replace("_","-")+"_"
    
    # Debugging    
    # config.nope = True
    
    # Gets dates from 1971-01-01 to 2022-12-31 with 3 hour separation
    # Entire 2023 year used as a val set 
    tdates = get_dates([(D(1971, 1, 1), D(2022, 12, 31), timedelta(hours=3))])
    
    # Gather mesh variables using NeoDatasetConfig (lives in utils.py)
    extra_out = ['tc-maxws', 'tc-minp'] # Cyclone max wind speed, Cyclone min pressure
    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=len(extra_out), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_out, input_levels=levels_medium, levels=levels_medium)
    
    timesteps = [6,24,72]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                     timesteps=timesteps,
                                     requested_dates=tdates,
                                     only_at_z=list(range(0, 24, 3))))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=False, 
            neorad=True, 
            window_size=(3,5,7)))

    #config.num_workers = 2
    config.prefetch_factor = 2
    #config.compute_Bcrit_every = 10
    config.ignore_train_safegaurd = True
    config.log_every = 10
    config.save_every = 100
    config.save_imgs_every = 1000
    config.HALF = True
    config.pin_memory = True
    
    config.gpus = '1-2'
    
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    
    config.lr_sched.cosine_en = True
    config.lr_sched.div_factor = 4
    config.lr_sched.step_offset = 0
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.cosine_period = 40_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.lr = 1e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    
    config.timeout = timedelta(minutes=20)
    
    w = WeatherTrainer(conf=config, model=model, data=data)
    w.run()

# Runs baseline TC model with extra variables (input and output)
# Also uses pure baseline (levels_joank) for output levels
#@launch(nodes={'baga': 1}, port=29501, start_method='spawn', kill_nvidia=False) # Debugging
@launch(nodes={'baga': 2}, port=29504, start_method="spawn")#, zulip=True, ping='@**Jack Michaels**')
def Nov21_TCinputoutput():
    # Resuming config
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    
    # Debugging    
    # config.nope = True
    
    # Gets dates from 1971-01-01 to 2022-12-31 with 3 hour separation
    # Entire 2023 year used as a val set 
    tdates = get_dates([(D(1971, 1, 1), D(2022, 12, 31), timedelta(hours=3))])
    
    # Gather mesh variables using NeoDatasetConfig (lives in utils.py)
    extra = ['tc-maxws', 'tc-minp'] # Cyclone max wind speed, Cyclone min pressure
    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    
    timesteps = [6,24,72]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                     timesteps=timesteps,
                                     requested_dates=tdates,
                                     only_at_z=list(range(0, 24, 3))))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,8,8),
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=False, 
            neorad=True, 
            window_size=(3,5,7)))

    #config.num_workers = 2
    config.prefetch_factor = 2
    config.compute_Bcrit_every = 10
    config.ignore_train_safegaurd = True
    config.log_every = 10
    config.save_every = 100
    config.save_imgs_every = 1000
    config.HALF = True
    config.pin_memory = True
    
    config.gpus = '3-4'
    
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    
    config.lr_sched.cosine_en = True
    config.lr_sched.div_factor = 4
    config.lr_sched.step_offset = 0
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.cosine_period = 40_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.lr = 1e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    
    config.timeout = timedelta(minutes=20)
    
    w = WeatherTrainer(conf=config, model=model, data=data)
    w.run()
    
# Runs basic training loop (with cyclone extra variables)
#@launch(ddp=False) # Debugging
#@launch(nodes={'baga': 5}, port=29500, start_method="spawn", kill_nvidia=False) # Debugging
@launch(ddp=False, nodes={'baga': 1}, port=29500, start_method="spawn", kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')
def Nov20_baselineTC():
    # Resuming config
    # config.resume = "_"+config.activity.replace("_","-")+"_"
    
    # Debugging    
    # config.nope = True
    
    # Gets dates from 1971-01-01 to 2022-12-31 with 3 hour separation
    # Entire 2023 year used as a val set 
    tdates = get_dates([(D(1971, 1, 1), D(2022, 12, 31), timedelta(hours=3))])
    
    # Gather mesh variables using NeoDatasetConfig (lives in utils.py)
    extra_out = ['tc-maxws', 'tc-minp'] # Cyclone max wind speed, Cyclone min pressure
    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=len(extra_out), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_out, input_levels=levels_medium, levels=levels_medium)
    
    timesteps = [6,24,72]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                     timesteps=timesteps,
                                     requested_dates=tdates,
                                     only_at_z=list(range(0, 24, 3))))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=False, 
            neorad=True, 
            window_size=(3,5,7)))

    #config.num_workers = 2
    config.prefetch_factor = 2
    #config.compute_Bcrit_every = 10
    config.ignore_train_safegaurd = True
    config.log_every = 10
    config.save_every = 100
    config.HALF = True
    config.pin_memory = True
    
    config.gpus = '0'
    
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    
    config.lr_sched.cosine_en = True
    config.lr_sched.div_factor = 4
    config.lr_sched.step_offset = 0
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.cosine_period = 40_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.lr = 1e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    
    config.timeout = timedelta(minutes=20)
    
    w = WeatherTrainer(conf=config, model=model, data=data)
    w.run()

# Runs basic training loop (with cyclone extra variables)
#@launch(ddp=False) # Debugging
@launch(nodes={'baga': 5}, start_method="spawn", zulip=True, ping='@**Jack Michaels**')
def Nov15_baselineTC():
    # Resuming config
    # config.resume = "_"+config.activity.replace("_","-")+"_"
    
    # Debugging    
    # config.nope = True
    
    # Gets dates from 1971-01-01 to 2023-12-31 with 3 hour separation
    tdates = get_dates([(D(1971, 1, 1), D(2023, 12, 31), timedelta(hours=3))])
    
    # Gather mesh variables using NeoDatasetConfig (lives in utils.py)
    extra_out = ['tc-maxws', 'tc-minp'] # Cyclone max wind speed, Cyclone min pressure
    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=len(extra_out), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_out, input_levels=levels_medium, levels=levels_medium)
    
    timesteps = [6,24,72]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                     timesteps=timesteps,
                                     requested_dates=tdates,
                                     only_at_z=list(range(0, 24, 3))))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=False, 
            neorad=True, 
            window_size=(3,5,7)))

    config.num_workers = 2
    config.prefetch_factor = 2
    config.compute_Bcrit_every = 10
    config.ignore_train_safegaurd = True
    config.log_every = 10
    config.save_every = 100
    config.HALF = True
    config.pin_memory = True
    
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    
    config.lr_sched.cosine_en = True
    config.lr_sched.div_factor = 4
    config.lr_sched.step_offset = 0
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.cosine_period = 12_500
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.lr = 1e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    
    config.timeout = timedelta(minutes=20)
    
    w = WeatherTrainer(conf=config, model=model, data=data)
    w.run()

# Runs basic training loop (no extra variables)
@launch(nodes={'baga': 5}, start_method="spawn", zulip=True, ping='@**Jack Michaels**')
def Nov12_basicnovars():
    
    # Gets dates from 2021-02-01 to 2024-06-01 with 1 day separation
    tdates = get_dates([(D(2021, 2, 1), D(2024, 6, 1))])
    
    # Gather mesh variables using NeoDatasetConfig (lives in utils.py)
    # Automatically loads from /fast/proc/ but can be specified to load from /jersey/ as well?
    mesh = meshes.LatLonGrid(source='era5-28', input_levels=levels_medium, levels=levels_medium)
    
    # Defaults to 24 (1 day increments)
    timesteps = [6, 12, 18] # Time steps used for prediction. Builds off of 6 hour forecast model
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                     timesteps=timesteps, 
                                     requested_dates=tdates)) # Can also specify only_at_z which is z times that we can use for data
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True, # Sinusoidal embeddings
            padded_lon=True, # Padded longitude
            Transformer=SlideLayer3D, # Swin Transformer
            checkpointfn=matepoint.checkpoint, # matepoint
            patch_size=(4,8,8), # Patch size for encoder convs; (4,8,8) is default
            hidden_dim=256, # Hidden size for transformer; 896 is default
            enc_swin_depth=6, # Encoder depth for transformer; 4 is default
            dec_swin_depth=6, # Decoder depth for transformer; 4 is default
            proc_swin_depth=6, # Like 99% sure this is processing swin depth (processing depth); 8 is default
            dims_per_head=32, # Dims per head transformer; 32 is default
            processor_dt=6, # Change in time for processor; 6 is default
            neorad=True,
            window_size=(3,5,7)
        )
    )
    
    config.timesteps = timesteps
    config.auto_loss_consts = True
    config.use_neoloader = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 10
    config.optim = 'shampoo'
    config.reset_optimizer = False
    config.lr_sched.cosine_period = 35_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 300
    config.lr_sched.lr = 8e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    w = WeatherTrainer(conf=config, model=model, data=data)
    w.run()

if __name__ == '__main__':
    run(locals().values())