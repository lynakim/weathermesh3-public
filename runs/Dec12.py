from launch import *

if am_i_torchrun():
    from train import *

from train import * # <--- comment this back in if not using DDP
from torch.utils import checkpoint as torch_checkpoint

@launch(nodes={'muir': 6},port=29505, start_method="spawn", zulip=True, ping='@**Haoxing Du**',validate=False,kill_nvidia=True)
def Oct7_rtyb_rand144hr_long():
    #torch.autograd.set_detect_anomaly(True)
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates((D(2021, 3, 20),D(2022, 7, 30)))

    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2], 
            outputs=[omesh], 
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
            processor_dt=[1,6], 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    
    #timesteps = [0,3,6,9,24,48]
    timesteps = list(range(49))
    data = WeatherDataset(DataConfig(inputs=[mesh1,mesh2], outputs=[omesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24)),
                                           random_timestep_subset = 6,
                                           ))
    
    import evals.package_neo as pn
    modelw = pn.get_yamahabachelor6()
    new_dict = model_state_dict_name_mapping(modelw.state_dict(),model) 
    def mapping(state_dict):
        new_state_dict = {}
        for k,v in state_dict.items():
            newk = k
            if k.startswith('encoder.'):
                new_state_dict[k.replace('encoder.','encoders.0.')] = v
                new_state_dict[k.replace('encoder.','encoders.1.')] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    new_dict = mapping(new_dict)
    model.load_state_dict(new_dict,strict=True) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.encoders.parameters():
        param.requires_grad = True
    
    config.HALF = True
    config.timesteps = timesteps
    config.ignore_train_safegaurd = True
    config.log_every = 5
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 2000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 250
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.print_ram_usage = True
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0)
#@launch(ddp=0,start_method='spawn')
@launch(nodes={'muir': 4},port=29505, start_method="spawn", zulip=True, ping='@**Haoxing Du**',validate=False,kill_nvidia=True)
def Sep26_YBft_rand144hr_long():
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    import evals.package_neo as pn
    model = pn.get_yamahabachelor0()
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    timesteps = [0] + list(range(1,25,1)) + list(range(28,49,4)) + list(range(54,97,6)) + list(range(108,145,12))
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                        timesteps=timesteps,
                                        max_ram_manual=int(4e9),
                                        worker_complain = False,
                                        requested_dates = tdates,
                                        only_at_z = list(range(24)),
                                        no_neoloader = True,
                                        random_timestep_subset = 6,
                                        ))
    config.timesteps = timesteps
    config.HALF = True
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 5
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 2048
    #config.optim = 'adam'
    config.reset_optimizer = False
    #config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 10_000
    config.lr_sched.cosine_bottom = 5e-9
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 5e-5
    config.use_neoloader = False
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.auto_loss_consts = True
    config.print_ram_usage = True
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'barceloneta': 6},port=29504, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Sep25_bachelor_hella_dts():

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',bro_zero_me_out=3,extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [0,6,12,18,24,36,48,72]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           random_timestep_subset = 4
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,8,8),                                       
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            neorad=True, 
            window_size=(5,7,7)))

    config.timesteps = timesteps
    config.auto_loss_consts = True
    config.use_neoloader = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.reset_optimizer = False
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1500
    config.lr_sched.lr = 0.3e-3 
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'barceloneta': 6},port=29505, start_method="spawn", zulip=True, ping='@**Haoxing Du**',validate=False,kill_nvidia=True)
def Sep24_YB_serp3rand144hr_rtft():
    #torch.autograd.set_detect_anomaly(True)
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates((D(2021, 3, 20),D(2022, 7, 30)))

    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25', bro_zero_me_out=3, extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='hres-13', bro_zero_me_out=3, extra_sfc_pad=3, input_levels=levels_tiny, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2], 
            outputs=[omesh], 
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
            processor_dt=[1,6], 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    
    timesteps = [0,3,6,9,24,48]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1,mesh2], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    
    import evals.package_neo as pn
    modelw = pn.get_yamahabachelor5()
    new_dict = model_state_dict_name_mapping(modelw.state_dict(),model) 
    def mapping(state_dict):
        new_state_dict = {}
        for k,v in state_dict.items():
            newk = k
            if k.startswith('encoder.'):
                new_state_dict[k.replace('encoder.','encoders.0.')] = v
                new_state_dict[k.replace('encoder.','encoders.1.')] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    new_dict = mapping(new_dict)
    model.load_state_dict(new_dict,strict=True) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.encoders.parameters():
        param.requires_grad = True
    
    config.HALF = True
    config.timesteps = timesteps
    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 2000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 250
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.print_ram_usage = True
    config.dates = tdates
    config.save_imgs = True
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 6},port=29504, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Sept23_bachelor_retest_noneolaoder():

    #config.gpus = '0-5'
    #config.nope = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',bro_zero_me_out=3,extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,8,8),                                       
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96, 
            neorad=True, 
            window_size=(3,5,7)))

    config.timesteps = timesteps
    config.use_neoloader = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.reset_optimizer = False
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 0.3e-3 
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(ddp=0)
def package_test():
    from evals.package_neo import get_bachelor
    get_bachelor()

@launch(nodes={'baga': 4},port=29505, start_method="spawn", zulip=True, ping='@**Haoxing Du**',validate=False,kill_nvidia=True)
#@launch(ddp=0)
def Sep20_YB_ft_serp3_rand144hr():
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    extra = ['logtp', '15_msnswrf', '45_tcc']
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    import evals.package_neo as pn

    timesteps = [0] + list(range(1,25,1)) + list(range(28,49,4)) + list(range(54,97,6)) + list(range(108,145,12))
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                        timesteps=timesteps,
                                        max_ram_manual=int(4e9),
                                        worker_complain = False,
                                        requested_dates = tdates,
                                        only_at_z = list(range(24)),
                                        no_neoloader = True,
                                        random_timestep_subset = 6,
                                        ))
    model = pn.get_yamahabachelor0()
    config.timesteps = timesteps
    config.HALF = True
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 5
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 2048
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 3500
    config.lr_sched.cosine_bottom = 5e-9
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.use_neoloader = False
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.auto_loss_consts = True
    config.print_ram_usage = True
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 6},port=29505, start_method="spawn", zulip=True, ping='@**Haoxing Du**',validate=False,kill_nvidia=True)
def Sep20_YB_rand144hr_rtft():
    #torch.autograd.set_detect_anomaly(True)
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates((D(2021, 3, 20),D(2022, 7, 30)))

    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny,levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2], 
            outputs=[omesh], 
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
            processor_dt=[1,6], 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    
    #timesteps = [3,6]
    timesteps = [0,3,6,9,24,48]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1,mesh2], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    
    import evals.package_neo as pn
    modelw = pn.get_yamahabachelor4()
    new_dict = model_state_dict_name_mapping(modelw.state_dict(),model) 
    def mapping(state_dict):
        new_state_dict = {}
        for k,v in state_dict.items():
            newk = k
            if k.startswith('encoder.'):
                new_state_dict[k.replace('encoder.','encoders.0.')] = v
                new_state_dict[k.replace('encoder.','encoders.1.')] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    new_dict = mapping(new_dict)
    model.load_state_dict(new_dict,strict=True) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.encoders.parameters():
        param.requires_grad = True
    
    config.HALF = True
    config.timesteps = timesteps
    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 2000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 250
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.print_ram_usage = True
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'baga': 4},port=29505, start_method="spawn", zulip=True, ping='@**Haoxing Du**',validate=False,kill_nvidia=True)
#@launch(ddp=0)
def Sep18_YB_ft_rand144hr():
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    extra = ['logtp', '15_msnswrf', '45_tcc']
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    import evals.package_neo as pn

    timesteps = list(range(1,25,1)) + list(range(28,49,4)) + list(range(54,97,6)) + list(range(108,145,12))
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                        timesteps=timesteps,
                                        max_ram_manual=int(4e9),
                                        worker_complain = False,
                                        requested_dates = tdates,
                                        only_at_z = list(range(24)),
                                        no_neoloader = True,
                                        random_timestep_subset = 6,
                                        ))
    model = pn.get_yamahabachelor3()
    config.timesteps = timesteps
    config.HALF = True
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 5
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 2048
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 2000
    config.lr_sched.cosine_bottom = 5e-9
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.use_neoloader = False
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.auto_loss_consts = True
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 6},port=29505, start_method="spawn", zulip=True, ping='@**Haoxing Du**',validate=False,kill_nvidia=True)
def Sep17_YB_rand96hr_rtft():
    #torch.autograd.set_detect_anomaly(True)
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates((D(2021, 3, 20),D(2022, 7, 30)))

    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny,levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2], 
            outputs=[omesh], 
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
            processor_dt=[1,6], 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    
    #timesteps = [3,6]
    timesteps = [3,6,9,24,48]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1,mesh2], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    
    import evals.package_neo as pn
    modelw = pn.get_yamahabachelor()
    new_dict = model_state_dict_name_mapping(modelw.state_dict(),model) 
    def mapping(state_dict):
        new_state_dict = {}
        for k,v in state_dict.items():
            newk = k
            if k.startswith('encoder.'):
                new_state_dict[k.replace('encoder.','encoders.0.')] = v
                new_state_dict[k.replace('encoder.','encoders.1.')] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    new_dict = mapping(new_dict)
    model.load_state_dict(new_dict,strict=True) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.encoders.parameters():
        param.requires_grad = True
    
    config.HALF = True
    config.timesteps = timesteps
    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 2000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 250
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'baga': 5},port=29505, start_method="spawn", zulip=True, ping='@**Haoxing Du**',validate=False,kill_nvidia=True)
#@launch(ddp=0)
def Sep16_YB_ft1hr24_4hr48_6hr96_rand():
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    extra = ['logtp', '15_msnswrf', '45_tcc']

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 31))])
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    import evals.package_neo as pn

    timesteps = list(range(1,25,1)) + list(range(28,49,4)) + list(range(54,97,6))
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                        timesteps=timesteps,
                                        max_ram_manual=int(4e9),
                                        worker_complain = False,
                                        requested_dates = tdates,
                                        only_at_z = list(range(24)),
                                        no_neoloader = True,
                                        random_timestep_subset = 6,
                                        ))
    model = pn.get_yamahabachelor()
    config.timesteps = timesteps
    config.HALF = True
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 5
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 2048
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 1500
    config.lr_sched.cosine_bottom = 5e-9
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.use_neoloader = False
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.auto_loss_consts = True
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 3},port=29505, start_method="spawn", zulip=True, ping='@**Haoxing Du**',validate=False,kill_nvidia=True)
def Sep16_YB_1hr24_rtft():
    #torch.autograd.set_detect_anomaly(True)
    config.gpus = '0-2'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates((D(2021, 3, 20),D(2022, 7, 30)))

    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny,levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2], 
            outputs=[omesh], 
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
            processor_dt=[1,6], 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    
    #timesteps = [3,6]
    timesteps = [3,6,9,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1,mesh2], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    
    import evals.package_neo as pn
    modelw = pn.get_yamahabachelor()
    new_dict = model_state_dict_name_mapping(modelw.state_dict(),model) 
    def mapping(state_dict):
        new_state_dict = {}
        for k,v in state_dict.items():
            newk = k
            if k.startswith('encoder.'):
                new_state_dict[k.replace('encoder.','encoders.0.')] = v
                new_state_dict[k.replace('encoder.','encoders.1.')] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    new_dict = mapping(new_dict)
    model.load_state_dict(new_dict,strict=True) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.encoders.parameters():
        param.requires_grad = True
    
    config.HALF = True
    config.use_neoloader = False
    config.timesteps = timesteps
    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 2000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 150
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.lr_sched.step_offset = 0
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()




#@launch(nodes={'muir': 4},port=29505, start_method="spawn")# zulip=True), ping='@**John Dean**',validate=True)#,kill_nvidia=True)
#@launch(nodes={'halfmoon':1},port=29505, start_method="spawn")# zulip=True), ping='@**John Dean**',validate=True)#,kill_nvidia=True)
@launch(ddp=0)
def Sep16_dataloader_testing():
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    extra = ['logtp', '15_msnswrf', '45_tcc']

    # tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 1, 1))])
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    import evals.package_neo as pn
    #timesteps = list(range(1,24,1))
    timesteps = [0,6,24,48,53]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                        timesteps=timesteps,
                                        max_ram_manual=int(4e9),
                                        worker_complain = False,
                                        requested_dates = tdates,
                                        only_at_z = list(range(24)),
                                        no_neoloader = True,
                                        #random_timestep_subset = 6,
                                        ))
    model = pn.get_yamahabachelor()
    config.timesteps = timesteps
    config.HALF = True
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 5
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 2048
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 1500
    config.lr_sched.cosine_bottom = 5e-9
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.use_neoloader = False
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.auto_loss_consts = True
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 4},port=29505, start_method="spawn")# zulip=True), ping='@**John Dean**',validate=True)#,kill_nvidia=True)
#@launch(ddp=0)
def Sep14_YB_ft1hr24_rand():
    #a=ForecastStep3D.simple_gen_todo([1,2,25],[1,6])
    #print(a)
    #return
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    extra = ['logtp', '15_msnswrf', '45_tcc']

    # tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 1, 1))])
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    import evals.package_neo as pn

    
    #timesteps = [0,1,2,3,5,6,7,9,11,12,13,17,18,19,23,24,25,27,]
    #timesteps = [0,1,2,3,5,6,7,9,11,12,13,17,18,19,23,24,25,27,]
    #timesteps = [0,1,2,4,6,7,12,18,24]
    #timesteps = [1,2,5,6,7,8,11,12,13,14,15,18,19,23,24]
    #timesteps = [1,2,12,13,132,133]
    #timesteps = [1,3,6]
    timesteps = list(range(1,24,1))
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                        timesteps=timesteps,
                                        max_ram_manual=int(4e9),
                                        worker_complain = False,
                                        requested_dates = tdates,
                                        only_at_z = list(range(24)),
                                        no_neoloader = True,
                                        random_timestep_subset = 6,
                                        ))
    model = pn.get_yamahabachelor()
    config.timesteps = timesteps
    config.HALF = True
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 5
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 2048
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 1500
    config.lr_sched.cosine_bottom = 5e-9
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.use_neoloader = False
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.auto_loss_consts = True
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'muir': 4},port=29505, start_method="spawn", zulip=True, ping='@**John Dean**',validate=True)#,kill_nvidia=True)
#@launch(ddp=0)
def Sep12_YB_fullfinetune_dataloading():
    #a=ForecastStep3D.simple_gen_todo([1,2,25],[1,6])
    #print(a)
    #return
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    extra = ['logtp', '15_msnswrf', '45_tcc']

    # tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 1, 1))])
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    import evals.package_neo as pn

    
    #timesteps = [0,1,2,3,5,6,7,9,11,12,13,17,18,19,23,24,25,27,]
    #timesteps = [0,1,2,3,5,6,7,9,11,12,13,17,18,19,23,24,25,27,]
    #timesteps = [0,1,2,4,6,7,12,18,24]
    #timesteps = [1,2,5,6,7,8,11,12,13,14,15,18,19,23,24]
    #timesteps = [1,2,12,13,132,133]
    #timesteps = [1,3,6]
    timesteps = list(range(1,150))
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                        timesteps=timesteps,
                                        max_ram_manual=int(4e9),
                                        worker_complain = False,
                                        requested_dates = tdates,
                                        only_at_z = list(range(24)),
                                        no_neoloader = True,
                                        random_timestep_subset = 6,
                                        ))
    model = pn.get_yamahabachelor()
    config.timesteps = timesteps
    config.HALF = True
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 5
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 2048
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 1000
    config.lr_sched.cosine_bottom = 5e-9
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.use_neoloader = False
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.auto_loss_consts = True
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 3},port=29505, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False,kill_nvidia=True)
#@launch(ddp=0)
def Sep13_YB_gfshres_ft():
    #torch.autograd.set_detect_anomaly(True)
    config.gpus = '0-2'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates((D(2021, 3, 20),D(2022, 7, 30)))

    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny,levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2], 
            outputs=[omesh], 
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
            processor_dt=[1,6], 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    
    #timesteps = [3,6]
    timesteps = [3,6,9,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1,mesh2], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    
    import evals.package_neo as pn
    modelw = pn.get_yamahabachelor()
    new_dict = model_state_dict_name_mapping(modelw.state_dict(),model) 
    def mapping(state_dict):
        new_state_dict = {}
        for k,v in state_dict.items():
            newk = k
            if k.startswith('encoder.'):
                new_state_dict[k.replace('encoder.','encoders.0.')] = v
                new_state_dict[k.replace('encoder.','encoders.1.')] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    new_dict = mapping(new_dict)
    model.load_state_dict(new_dict,strict=True) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.encoders.parameters():
        param.requires_grad = True
    
    config.HALF = True
    config.timesteps = timesteps
    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 2000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 150
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 4},port=29505, start_method="spawn", zulip=True, ping='@**John Dean**',validate=True)#,kill_nvidia=True)
def Sep12_YB_fullfinetune():
    #a=ForecastStep3D.simple_gen_todo([1,2,25],[1,6])
    #print(a)
    #return
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    extra = ['logtp', '15_msnswrf', '45_tcc']

    # tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 1, 1))])
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    import evals.package_neo as pn

    timesteps = [1,7,12,14,18,25,49,96,99]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                        timesteps=timesteps, max_ram_manual=int(4e9),
                                        worker_complain = False,
                                        requested_dates = tdates,
                                        only_at_z = list(range(24))
                                        ))
    
    model = pn.get_yamahabachelor()
    config.timesteps = timesteps
    config.HALF = True
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 2048
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 1000
    config.lr_sched.cosine_bottom = 5e-9
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.auto_loss_consts = True
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()







def Sep7_doctorate(): # copied this in from Nov12.py
    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    config.disregard_buffer_checksum = False#True
    #config.nope = True
    # tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 5, 1))])
    extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    # imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra_all, output_only_vars=extra_out_only, input_levels=levels_medium, levels=levels_joank)
    # omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra_all, output_only_vars=extra_out_only, input_levels=levels_medium, levels=levels_joank)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28', extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28', extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24, 36]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           max_instance_queue_size=6,
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,8,8), 
            hidden_dim=1536, #1408, 
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=3, 
            output_deltas=False, 
            decoder_reinput_initial=False,
            neorad=True, 
        window_size=(3,5,7)))

    config.HALF = True
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 2048
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    # Scaling by sqrt(3) now that we're using 
    config.lr_sched.lr = 0.3e-3 
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.loss_consts = {48: 0.3, 24: 1.0, 6: 1.0, 3: 1.0, 36: 0.25}
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0)
@launch(nodes={'stinson': 4},port=29505, start_method="spawn")#, zulip=True, ping='@**John Dean**',validate=True)#,kill_nvidia=True)
def Sep4_s2s_cluelessyolo():
    config.gpus = '0-3'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 5, 1))])
    extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra_all, output_only_vars=extra_out_only, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra_all, output_only_vars=extra_out_only, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24,24*6,24*12,24*24]
    #timesteps = [1,3,6,9]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(0,24,3))
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,16,16), 
            hidden_dim=512, 
            enc_swin_depth=8, 
            dec_swin_depth=8, 
            proc_swin_depth=8, 
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=24, 
            neorad=True,
        window_size=(3,5,7)))
    
    yield model    
    config.loss_consts = {24: 1.0, 24*6: 1, 24*12: 0.5, 24*24: 0.25}
    config.HALF = True
    config.ignore_train_safegaurd = True
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 10_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 300
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 250
    config.lr_sched.lr = 3e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0)
@launch(nodes={'baga': 6},port=29505, start_method="spawn")#, zulip=True, ping='@**John Dean**',validate=False,kill_nvidia=True)
def Sep4_s2s_stupidyolo():
    #config.gpus = '0-2'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_vars=extra, output_only_vars=extra, is_output_only=True, input_levels=levels_gfs, levels=levels_joank)
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_pad=3, is_output_only=True, input_levels=levels_gfs, levels=levels_joank)
    #mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='hres-13', extra_sfc_pad=3, is_output_only=True,  input_levels=levels_tiny,levels=levels_joank)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24,24*6,24*12,24*24]
    #timesteps = [1,3,6,9]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,16,16), 
            hidden_dim=1024, 
            enc_swin_depth=8, 
            dec_swin_depth=8, 
            proc_swin_depth=8, 
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=24, 
            neorad=True,
        window_size=(3,5,7)))
    
    yield model    
    config.loss_consts = {24: 1.0, 24*6: 1, 24*12: 0.5, 24*24: 0.25}
    config.HALF = True
    config.ignore_train_safegaurd = True
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 15000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 300
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 250
    config.lr_sched.lr = 3e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'baga': 3},port=29505, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False,kill_nvidia=True)
#@launch(ddp=0)
def Aug28_bachelor_gfshres_ft_serp3_2k():
    config.gpus = '0-2'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates((D(2021, 3, 20),D(2022, 7, 30)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_vars=extra, output_only_vars=extra, is_output_only=True, input_levels=levels_gfs, levels=levels_joank)
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_pad=3, is_output_only=True, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='hres-13', extra_sfc_pad=3, is_output_only=True,  input_levels=levels_tiny,levels=levels_joank)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24]
    #timesteps = [1,3,6,9]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1,mesh2], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=torch_checkpoint.checkpoint,
            patch_size=(5,8,8), 
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    
    yield model
    
    lpath = '/huge/deep/runs/run_Aug25-serpentbachelor3_20240825-192152/model_epoch1_iter93594_step15599_loss0.039.pt'
    checkpoint = torch.load(lpath,map_location='cpu')
    #for k in list(checkpoint['model_state_dict'].keys()):
    #    if k.startswith('proc_swin'):
    #        del checkpoint['model_state_dict'][k]
    #    else: assert 'proc_swin' not in k
    new_dict = model_state_dict_name_mapping(checkpoint['model_state_dict'],model) 
    def mapping(state_dict):
        new_state_dict = {}
        for k,v in state_dict.items():
            newk = k
            if k.startswith('encoder.'):
                new_state_dict[k.replace('encoder.','encoders.0.')] = v
                new_state_dict[k.replace('encoder.','encoders.1.')] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    new_dict = mapping(new_dict)

    model.load_state_dict(new_dict,strict=True) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.encoders.parameters():
        param.requires_grad = True
    
    #config.loss_consts = {1: 1.0, 3: 0.4, 6: 0.2, 12: 0.1}
    #config.reset_steps_on_resume = True
    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 2000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 150
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'baga': 3},port=29505, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False,kill_nvidia=False)
#@launch(ddp=0)
def Aug28_bachelor_hres_ft():
    config.gpus = '3-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates((D(2021, 3, 20),D(2022, 7, 30)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_vars=extra, output_only_vars=extra, is_output_only=True, input_levels=levels_gfs, levels=levels_joank)
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_pad=3, is_output_only=True, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='hres-13', extra_sfc_pad=3, is_output_only=True,  input_levels=levels_tiny,levels=levels_joank)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24]
    #timesteps = [1,3,6,9]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh2], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=torch_checkpoint.checkpoint,
            patch_size=(5,8,8), 
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True, 
        window_size=(3,5,7)))
    
    print(model)
    yield model
    
    lpath = '/huge/deep/runs/run_Aug11-serpentbachelor2_20240801-205915/model_epoch1_iter191994_step31999_loss0.062.pt'
    checkpoint = torch.load(lpath,map_location='cpu')
    #for k in list(checkpoint['model_state_dict'].keys()):
    #    if k.startswith('proc_swin'):
    #        del checkpoint['model_state_dict'][k]
    #    else: assert 'proc_swin' not in k
    new_dict = model_state_dict_name_mapping(checkpoint['model_state_dict'],model)
    model.load_state_dict(new_dict,strict=True) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    #config.loss_consts = {1: 1.0, 3: 0.4, 6: 0.2, 12: 0.1}
    #config.reset_steps_on_resume = True
    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 4_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 250
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0)
@launch(nodes={'baga': 3},port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False,kill_nvidia=True)
def Aug28_bachelor_gfs_ft():
    config.gpus = '0-2'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates((D(2021, 3, 20),D(2022, 7, 30)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_vars=extra, output_only_vars=extra, is_output_only=True, input_levels=levels_gfs, levels=levels_joank)
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_pad=3, is_output_only=True, input_levels=levels_gfs, levels=levels_joank)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24]
    #timesteps = [1,3,6,9]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=torch_checkpoint.checkpoint,
            patch_size=(5,8,8), 
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True, 
        window_size=(3,5,7)))
    
    yield model
    
    lpath = '/huge/deep/runs/run_Aug11-serpentbachelor2_20240801-205915/model_epoch1_iter191994_step31999_loss0.062.pt'
    checkpoint = torch.load(lpath,map_location='cpu')
    #for k in list(checkpoint['model_state_dict'].keys()):
    #    if k.startswith('proc_swin'):
    #        del checkpoint['model_state_dict'][k]
    #    else: assert 'proc_swin' not in k
    model.load_state_dict(checkpoint['model_state_dict'],strict=False) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.enc_swin.parameters():
        param.requires_grad = True
    for param in model.conv.parameters():
        param.requires_grad = True
    for param in model.conv_sfc.parameters():
        param.requires_grad = True
    
    #config.loss_consts = {1: 1.0, 3: 0.4, 6: 0.2, 12: 0.1}
    #config.reset_steps_on_resume = True
    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 4_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 250
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 0.5e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0)
@launch(nodes={'ip-172-31-2-177': 8},port=29500, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
def Sep4_cirrus_honda():
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_joank, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_joank, levels=levels_joank)
    #timesteps = [6, 24]
    timesteps = [1,3,6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(100e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=torch_checkpoint.checkpoint,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            timesteps=timesteps,
            dims_per_head=32,
            processor_dt=1, 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True, 
        window_size=(3,5,7)))
    
    yield model
    
    #lpath = '/huge/deep/runs/run_Aug11-serpentbachelor2_20240801-205915/model_epoch1_iter191994_step31999_loss0.062.pt'
    lpath = '/huge/deep/runs/run_Aug25-serpentbachelor3_20240825-192152/model_epoch1_iter93594_step15599_loss0.039.pt'

    checkpoint = torch.load(lpath,map_location='cpu')
    for k in list(checkpoint['model_state_dict'].keys()):
        if k.startswith('proc_swin'):
            del checkpoint['model_state_dict'][k]
        else: assert 'proc_swin' not in k
    new_dict = model_state_dict_name_mapping(checkpoint['model_state_dict'],model)
    model.load_state_dict(new_dict,strict=False) 
    #for param in model.parameters():
    #    param.requires_grad = False
    #
    #for param in model.proc_swin.parameters():
    #    param.requires_grad = True

    config.loss_consts = {1: 1.0, 3: 0.4, 6: 0.2}
    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 30_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 3e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0)
#@launch(nodes={'singing': 4},port=29500, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
@launch(nodes={'muir': 6},port=29500, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
def Sep6_yamaha_bigsad():
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    config.reset_optimizer = False
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    #timesteps = [6, 24]
    timesteps = [1,3,6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=torch_checkpoint.checkpoint,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            timesteps=timesteps,
            dims_per_head=32,
            processor_dt=1, 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True, 
        window_size=(3,5,7)))
    
    yield model
    
    #lpath = '/huge/deep/runs/run_Aug11-serpentbachelor2_20240801-205915/model_epoch1_iter191994_step31999_loss0.062.pt'
    lpath = '/huge/deep/runs/run_Aug25-serpentbachelor3_20240825-192152/model_epoch1_iter93594_step15599_loss0.039.pt'

    checkpoint = torch.load(lpath,map_location='cpu')
    for k in list(checkpoint['model_state_dict'].keys()):
        if k.startswith('proc_swin'):
            del checkpoint['model_state_dict'][k]
        else: assert 'proc_swin' not in k
    new_dict = model_state_dict_name_mapping(checkpoint['model_state_dict'],model)
    model.load_state_dict(new_dict,strict=False) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.proc_swin.parameters():
        param.requires_grad = True

    config.loss_consts = {1: 1.0, 3: 0.4, 6: 0.2}
    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 25
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 30_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 3e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0)
@launch(nodes={'ip-172-31-2-177': 8},port=29500, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
def Aug30_cirrus_yamaha():
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_joank, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_joank, levels=levels_joank)
    #timesteps = [6, 24]
    timesteps = [1,3,6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(100e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=torch_checkpoint.checkpoint,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            timesteps=timesteps,
            dims_per_head=32,
            processor_dt=1, 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True, 
        window_size=(3,5,7)))
    
    yield model
    
    #lpath = '/huge/deep/runs/run_Aug11-serpentbachelor2_20240801-205915/model_epoch1_iter191994_step31999_loss0.062.pt'
    lpath = '/huge/deep/runs/run_Aug25-serpentbachelor3_20240825-192152/model_epoch1_iter93594_step15599_loss0.039.pt'

    checkpoint = torch.load(lpath,map_location='cpu')
    for k in list(checkpoint['model_state_dict'].keys()):
        if k.startswith('proc_swin'):
            del checkpoint['model_state_dict'][k]
        else: assert 'proc_swin' not in k
    new_dict = model_state_dict_name_mapping(checkpoint['model_state_dict'],model)
    model.load_state_dict(new_dict,strict=False) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.proc_swin.parameters():
        param.requires_grad = True

    config.loss_consts = {1: 1.0, 3: 0.4, 6: 0.2}
    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 30_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 3e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0)
@launch(nodes={'baga': 6},port=29504, start_method="spawn", zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Aug28_yamaha():
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    #timesteps = [6, 24]
    timesteps = [1,3,6,9]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(12e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=torch_checkpoint.checkpoint,
            patch_size=(5,8,8), 
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=6, 
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=1, 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True, 
        window_size=(3,5,7)))
    
    yield model
    
    lpath = '/huge/deep/runs/run_Aug11-serpentbachelor2_20240801-205915/model_epoch1_iter191994_step31999_loss0.062.pt'
    checkpoint = torch.load(lpath,map_location='cpu')
    for k in list(checkpoint['model_state_dict'].keys()):
        if k.startswith('proc_swin'):
            del checkpoint['model_state_dict'][k]
        else: assert 'proc_swin' not in k
    model.load_state_dict(checkpoint['model_state_dict'],strict=False) 
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.proc_swin.parameters():
        param.requires_grad = True

    config.loss_consts = {1: 1.0, 3: 0.4, 6: 0.2, 12: 0.1}
    config.reset_steps_on_resume = True
    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 12_500
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 3e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'bimini': 6},port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
def Aug25_serpentbachelor3():
    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1999, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    #timesteps = [6, 24]
    timesteps = [6,24,72,144]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
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
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True, 
        window_size=(3,5,7)))

    yield model

    config.reset_steps_on_resume = True
    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 12_500
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'bimini': 6},port=29504, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Aug25_serpentbachelor2():
    import natten
    #assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,24,72,144]
    train_timesteps = [6, 24, 72, 144]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=train_timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8), 
                                                  hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, train_timesteps=train_timesteps, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,7)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = True
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 25_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    # Scaling by sqrt(3) now that we're using 
    w.preserved_conf.lr_sched.lr = 0.075e-3 
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()



@launch(nodes={'barceloneta': 6} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=True)
def Aug19_master48ft_JD():
    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [3, 6, 24, 48]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,8,8), 
            hidden_dim=1280, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=6, 
            lat_compress=False, 
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=3, 
            use_matepoint=True, 
            output_deltas=False, 
            decoder_reinput_initial=False,
            neorad=True, 
        window_size=(3,5,7)))

    yield model

    config.HALF = True
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    #config.optim = 'adam'
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 300
    config.lr_sched.lr = 0.3e-3  / 1.4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.loss_consts = {48: 0.4, 24: 1.0, 6: 1.0, 3: 1.0}
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'halfmoon': 1},port=29504, start_method="spawn")#, zulip=True, ping='@**Haoxing Du**')
def Aug19_serpentprop():
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [3, 6, 24, 48]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,8,8), 
            hidden_dim=1280, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=6, 
            lat_compress=False, 
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=3, 
            use_matepoint=True, 
            output_deltas=False, 
            decoder_reinput_initial=False,
            neorad=True, 
        window_size=(3,5,7)))

    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.validate_every = -1
    config.DH = 24
    config.log_every = 2
    config.save_every = 200
    config.optim = 'adam'
    config.shampoo.dim = 512 
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 3e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    config.val_dates = vdates
    config.serpent_backprop = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


#@launch(nodes={'bimini': 6},port=29504, start_method="spawn")
@launch(ddp=1,nodes={'halfmoon':4},start_method='spawn',zulip=True,ping='@**John Dean**')
#@launch(ddp=1,nodes={'halfmoon':4},start_method='spawn',zulip=True)
def Aug_6_cleantesting():
    import natten
    assert natten.DO_WRAP == True

    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    #tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    tdates = get_dates([(D(2019, 1, 23), D(2019, 1, 28))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStep3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,6,6), 
                                                  hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,5)))


    config.HALF = False
    config.ignore_train_safegaurd = True
    config.optim = 'adam'
    config.lr_sched.lr = 0.25e-3
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()





#@launch(nodes={'miramar': 6, 'bimini.fast': 6}, start_method="spawn")
@launch(nodes={'stinson': 4},port=29501, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Aug6_slidetest():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-3'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.resume = "/huge/deep/runs/run_Jul24-slidetest_20240724-172914/model_epoch1_iter103992_step12999_loss0.082.pt"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,24]
    #timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,6,6), 
                                                  hidden_dim=640, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    #w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    w.preserved_conf.lr_sched.restart_warmup_end_step = 1000

    w.preserved_conf.lr_sched.lr = 0.25e-3 #* 1.4
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(ddp=0,start_method='spawn')
def Jul30_nimbusnoddp():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    #extra = ['logtp', '15_msnswrf', '45_tcc']
    extra = []
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,24]
    #timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,6,6), 
                                                  hidden_dim=640, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    #w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    # Scaling by sqrt(3) now that we're using 
    w.preserved_conf.lr_sched.lr = 0.25e-3 
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'bimini': 6},port=29504, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Jul28_serpent():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,12,24,48,72]
    #timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,6,6), 
                                                  hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    # Scaling by sqrt(3) now that we're using 
    w.preserved_conf.lr_sched.lr = 0.25e-3 
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

#@launch(nodes={'stinson': 4, 'singing.fast': 4},port=29504, start_method="spawn")
@launch(nodes={'miramar': 6, 'bimini.fast': 6},port=29504, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Jul28_slide624ND():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,24]
    #timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,6,6), 
                                                  hidden_dim=640, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    # Scaling by sqrt(3) now that we're using 
    w.preserved_conf.lr_sched.lr = 0.25e-3 
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'singing': 4 },port=29504, start_method="spawn")
def Jul25_slide_nodeltas():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-3'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24,72]
    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8), 
                                                  hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=128, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    # Scaling by sqrt(3) now that we're using 
    w.preserved_conf.lr_sched.lr = 0.25e-3 
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

#@launch(nodes={'miramar': 6, 'bimini.fast': 6}, start_method="spawn")
@launch(nodes={'stinson': 4, 'singing.fast': 4},port=29504, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Jul24_slidetest():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-3'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.resume = "/huge/deep/runs/run_Jul24-slidetest_20240724-172914/model_epoch1_iter103992_step12999_loss0.082.pt"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,24]
    #timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,6,6), 
                                                  hidden_dim=640, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    #w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000

    w.preserved_conf.lr_sched.lr = 0.25e-3 #* 1.4
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'miramar': 6, 'bimini': 6},port=29504, start_method="spawn")
#@launch(nodes={'miramar': 6},port=29504, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Jul19_wrap_rpb_epb_fp32():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24,72]
    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8), 
                                                  hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=128, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    # Scaling by sqrt(3) now that we're using 
    w.preserved_conf.lr_sched.lr = 0.25e-3 
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'singing': 4},port=29504, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Jul19_wrap_rpb_fp32():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB turned on in model_latlon_3d.py!!

    config.gpus = '0-3'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24,72]
    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8), 
                                                  hidden_dim=1024, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=128, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))
    print(model)
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 1000000000
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    w.preserved_conf.lr_sched.lr = 0.25e-3 * 8/6 / 2
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'barceloneta': 4},port=29504, start_method="spawn")
def Jul19_wrap_NOpb_fp32():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### POSITION BIASES TURNED OFF IN model_latlon_3d.py!!

    config.gpus = '0-3'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24,72]
    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8), 
                                                  hidden_dim=1024, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=128, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    w.preserved_conf.lr_sched.lr = 0.25e-3 * 8/6 / 2
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'miramar': 4},port=29504, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Jul19_nowrap_NOpb_fp32():
    import natten
    assert natten.DO_WRAP == False
    config.gpus = '0-3'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24,72]
    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8), 
                                                  hidden_dim=1024, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=128, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    w.preserved_conf.lr_sched.lr = 0.25e-3 * 8/6 / 2
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'ip-172-31-11-78':8},port=29500, start_method="spawn",ddp=1)
def May1_shallowpony_cloud():
    #config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra = []
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    #omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra, is_output_only=True,input_levels=levels_medium, levels=levels_joank)
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, output_only_vars=extra, is_output_only=True,levels=levels_joank)


    #timesteps = [6,24]
    timesteps = [6,24,72]
    timesteps = [24,72]
    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps,
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           max_ram_manual = 150 * 2**30
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(5,8,8), hidden_dim=1504, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=4, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=128, neorad=True, window_size=(2,3,6)))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 10
    w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    w.preserved_conf.lr_sched.lr = (1/np.e)*1e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(ddp=0,start_method='spawn')
def May1_data():
    tdates = get_dates((D(2001,12,1), D(2019, 12, 2)))
    extra = []
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    timesteps = [1]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps,
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24)),
                                           use_mmap = False, # WAS TRUE BUT I SET IT TO FALSE SO WE COULD DO DDP TESTS
                                           num_workers = 20,
                                           max_ram_manual = 150 * 2**30
                                           ))

    data_loader = NeoLoader(data)
    while 1:
        time.sleep(10)
        #print("yo...")
                                        



@launch(ddp=0,nodes={'halfmoon':1},start_method='spawn')
def Jan16_adaptdev():
    #config.gpus = '0'
    config.nope = 1
    config.prefix = "_adapt"
    #config.name+= "-l1"
    #config.resume = "_"+config.name.replace("_","-")+"_"
    #config.resume = "run_Dec26-Luigi-l2_20231226-151561"

    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='gfs-28')
    #mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1],
                                           outputs = [mesh3],
                                           timesteps=[0,24],
                                           requested_dates = get_dates((D(2021, 3, 20),D(2022, 7, 30))),
                                           ))
    import model_unet
 
    model_conf = ForecastStepConfig(data.config.inputs, 
                outputs = data.config.outputs,
                patch_size=(4,8,8), 
                hidden_dim=768, 
                enc_swin_depth=0,
                dec_swin_depth=0, 
                proc_swin_depth=0, 
                adapter_swin_depth=8,
                timesteps=[0,24], 
                output_deltas = True,
                use_matepoint = False,
                processor_dt = -1,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True),
                )
    
    adatper = model_unet.ForecastStepAdapterLinear(model_conf)
    from evals.package_neo import get_neoquadripede
    forecaster = get_neoquadripede()
    model = model_unet.ForecastStepAdapterCombo(adatper,forecaster)
    # print number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1e6:0.1f}M")

    config.batch_size = 1
    config.use_neoloader = True
    config.ignore_train_safegaurd = True
    config.l2_loss = False
    config.validate_every = 100
    config.log_every = 20
    config.save_every = 1000
    config.val_date_range =  [D(2022, 8, 1), D(2023, 1, 7)]
    config.lr_sched.cosine_period = 30_000
    config.lr_sched.warmup_end_step = 1_000
    config.lr_sched.div_factor = 2
    config.lr_sched.lr = 3.5e-4
    config.optim = 'adam'#'shampoo'
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.rerun_sample = 1
    config.yolo = False

    w = WeatherTrainer(data,conf=config,model=model)
    w.run()

@launch(ddp=False)
def Jan13dev():
    tdates = get_dates((D(2019,12,1), D(2019, 12, 2)))
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra = []
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    timesteps = [1]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24)),
                                           use_mmap = True
                                           ))
    data.check_for_dates()
    sample = default_collate([data[0]])
    #select_bbox(sample[0][0],mesh,(0,0,5,10))
    pass


#@launch(ddp=0,port=29500, start_method="fork")
@launch(nodes={'ip-172-31-11-78':8},port=29500, start_method="spawn",ddp=1)
def cloudtest():
    #config.gpus = '0-3' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    #tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    tdates = get_dates((D(2001,12,1), D(2019, 12, 2)))
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra = []
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    timesteps = [1]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps,
                                           worker_complain = True,
                                           quiet = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24)),
                                           use_mmap = False, # WAS TRUE BUT I SET IT TO FALSE SO WE COULD DO DDP TESTS
                                           num_workers = 4,
                                           max_ram_manual = 150 * 2**30))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=True, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64))
     
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'adam'
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 20_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.33e-3
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(ddp=False)
def cloudstream():
    config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    mesh = meshes.LatLonGrid(CLOUD=1,source='era5-28',extra_sfc_vars=['logtp','45_tcc'])
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[24], max_ram_manual=int(6e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    data.check_for_dates()
    sample = data[0]
    imshow_compare(sample[0][0],mesh,sample[1][0],mesh,var='167_2t')

@launch(ddp=False)
def Raaam():
    config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[24], max_ram_manual=int(6e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=1024, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[24], dims_per_head=32, processor_dt=6, use_matepoint=True, output_deltas=False))
    data.check_for_dates()
    sample = default_collate([data[0]])
    model.to('cuda')
    with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        x = sample[0]
        x = [xx.to('cuda') for xx in x]
        y = model(x)
    

@launch(ddp=0)
def datatest():
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='gfs-28')
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_medium,levels=levels_tiny,source='era5-28',extra_sfc_vars=['logtp'])
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1, mesh2],
                                           outputs = [mesh3],
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2021, 3, 20),D(2022, 7, 30))),
                                           ))
    data.check_for_dates()
    sample = data[0]

    print(" | ".join([", ".join([f'({",".join([str(xx) for xx in x.shape])})' for x in t[:-1]])+", "+get_date_str(t[-1]) for t in sample]))
    x = sample[1][0]
    imshow_compare(x,mesh3,x,mesh3,var='logtp')
    print("yo")

    


@launch(ddp=1,nodes={'martins':3})
def logtest():
    print("youyoyo whatup")

@launch(ddp=1,nodes={'miramar':5},start_method='spawn')
#@launch(ddp=0,start_method='spawn')
def Dec27_Waluigi():
    #config.gpus = '1,2,3,5'
    #config.nope = 1
    config.prefix = "_adapt"
    #config.name += "-yolked3"
    #config.resume = "_"+config.name.replace("_","-")+"_"
    config.resume = 'run_Dec27-Waluigi_20231227-212234'
    #config.resume = "run_Dec27-Wario_"

    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='gfs-28')
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1, mesh2],
                                           outputs = [mesh3],
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2021, 3, 20),D(2022, 7, 30))),
                                           ))

    import model_unet
    model_conf = ForecastStepConfig(data.config.inputs, 
                outputs = data.config.outputs,
                patch_size=(4,8,8), 
                hidden_dim=768, 
                enc_swin_depth=0,
                dec_swin_depth=0,
                proc_swin_depth=0,
                adapter_swin_depth=8,
                timesteps=[0], 
                output_deltas = True,
                use_matepoint = False,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True)
                )
    
    model = model_unet.ForecastStepAdapterConv(model_conf)
    # print number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1e6:0.1f}M")

    config.batch_size = 1
    config.use_neoloader = True
    config.ignore_train_safegaurd = True
    config.l2_loss = False
    config.validate_every = 20
    config.log_every = 50
    config.save_every = 500
    config.val_date_range =  [D(2022, 8, 1), D(2023, 1, 7)]
    config.lr_sched.cosine_period = 20_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.div_factor = 2
    config.lr_sched.lr = 7.5e-4 * 5/6
    config.optim = 'shampoo' if am_i_torchrun() else 'adam' 
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.rerun_sample = 1
    config.yolo = False
    config.skip_audit = True

    w = WeatherTrainer(data,conf=config,model=model)
    w.run()


@launch(ddp=1,nodes={'stinson':4},start_method='spawn')
#@launch(ddp=1,nodes={'martins':1},start_method='spawn')
def Dec27_Wario():
    #config.gpus = '2,3'
    config.nope = 0
    config.prefix = "_adapt"
    config.name += "-4g-GELU"
    #config.resume = "_"+config.name.replace("_","-")+"_"
    #config.resume = "run_Dec27-Wario_"

    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='gfs-28')
    #mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1],
                                           outputs = [mesh3],
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2021, 3, 20),D(2023, 4, 30))),
                                           ))
    import model_unet
    model_conf = ForecastStepConfig(data.config.inputs, 
                outputs = data.config.outputs,
                patch_size=(4,8,8), 
                hidden_dim=768, 
                enc_swin_depth=0,
                dec_swin_depth=0, 
                proc_swin_depth=0, 
                adapter_swin_depth=8,
                timesteps=[0],
                activation = nn.GELU(),
                output_deltas = True,
                use_matepoint = False )
    
    model = model_unet.ForecastStepAdapterConv(model_conf)
    # print number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1e6:0.1f}M")

    config.batch_size = 1
    config.use_neoloader = True
    config.ignore_train_safegaurd = True
    config.l2_loss = False
    config.validate_every = 20
    config.log_every = 50
    config.save_every = 500
    config.val_date_range =  [D(2022, 5, 1), D(2023, 8, 31)]
    config.lr_sched.cosine_period = 20_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.div_factor = 2
    config.lr_sched.lr = 1e-3
    config.optim = 'shampoo' if am_i_torchrun() else 'adam'
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.rerun_sample = 1
    config.yolo = False
    config.skip_audit = True
    config.disregard_buffer_checksum = True

    w = WeatherTrainer(data,conf=config,model=model)
    w.run()

@launch(ddp=1,nodes={'stinson':4},start_method='spawn')
#@launch(ddp=1,nodes={'martins':1},start_method='spawn')
def Dec26_Luigi():
    #config.gpus = '0'
    config.nope = 0
    config.prefix = "_adapt"
    config.name+= "-l1"
    #config.resume = "_"+config.name.replace("_","-")+"_"
    config.resume = "run_Dec26-Luigi-l2_20231226-151561"

    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='gfs-28')
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1, mesh2],
                                           outputs = [mesh3],
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2021, 3, 20),D(2022, 7, 30))),
                                           ))
    import model_unet
 
    model_conf = ForecastStepConfig(data.config.inputs, 
                outputs = data.config.outputs,
                patch_size=(4,8,8), 
                hidden_dim=768, 
                enc_swin_depth=0,
                dec_swin_depth=0, 
                proc_swin_depth=0, 
                adapter_swin_depth=8,
                timesteps=[0], 
                output_deltas = True,
                use_matepoint = False,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True),
                )
    
    model = model_unet.ForecastStepAdapterConv(model_conf)
    # print number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1e6:0.1f}M")

    config.batch_size = 1
    config.use_neoloader = True
    config.ignore_train_safegaurd = True
    config.l2_loss = False
    config.validate_every = 100
    config.log_every = 20
    config.save_every = 1000
    config.val_date_range =  [D(2022, 8, 1), D(2023, 1, 7)]
    config.lr_sched.cosine_period = 30_000
    config.lr_sched.warmup_end_step = 1_000
    config.lr_sched.div_factor = 2
    config.lr_sched.lr = 3.5e-4
    config.optim = 'shampoo'
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.rerun_sample = 1
    config.yolo = True

    w = WeatherTrainer(data,conf=config,model=model)
    w.run()



#@launch(nodes={'stinson.fast': 4, 'halfmoon': 4},port=29500, start_method="spawn")
@launch(nodes={'miramar':4},port=29500, start_method="spawn")
def Dec22_neoenc():
    config.gpus = '1,2,3,5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[24], max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(1979, 1, 23),D(2017, 12, 28))),
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, 
                                                  lat_compress=False, timesteps=[24], dims_per_head=32, processor_dt=12,use_matepoint=False))
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    #w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 20_000
    w.preserved_conf.lr_sched.lr = 0.3e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 1, 23),D(2017, 12,28)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 28)))
    w.run()


@launch(ddp=False,nodes={'martins':1})
def Dec12_GFS_ERA5():
    config.gpus = '0'
    config.nope = False
    config.prefix = "_adapt"
    dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,levels=levels_tiny ,source='hres')
    dsc2 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc1,
                                           output_mesh = meshes.LatLonGrid(config=dsc2),
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2022, 1, 5),D(2023, 8, 25))),
                                           odds_idc = 1,
                                           use_mmap = True
                                           ))
    data.check_for_dates()
    model = ForecastStepAdapter2(ForecastStepConfig(data.config.mesh, 
                                                  output_mesh = data.config.output_mesh,
                                                  patch_size=(2,8,8), 
                                                  hidden_dim=768, 
                                                  enc_swin_depth=8,
                                                  dec_swin_depth=0, 
                                                  proc_swin_depth=0, 
                                                  timesteps=[0], 
                                                  dims_per_head=32,
                                                  output_deltas = False))
    
    w = WeatherTrainer(data,conf=config,model=model)

    w.preserved_conf.use_neoloader = False
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.log_every = 5
    w.preserved_conf.l2_loss = False
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = False
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 10_000
    w.preserved_conf.lr_sched.cosine_bottom = None
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 1e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    w.run()


@launch(ddp=False,nodes={'miramar':1})
def Dec19_linear():
    config.gpus = '0'
    config.nope = 0
    config.prefix = "_adapt"
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
    dsc2 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc1,
                                           output_mesh = meshes.LatLonGrid(config=dsc2),
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2022, 1, 5),D(2023, 8, 25))),
                                           odds_idc = 1,
                                           use_mmap = True
                                           ))
    data.check_for_dates()
    model = LinearAdapter(ForecastStepConfig(data.config.mesh, 
                    output_mesh = data.config.output_mesh,
                    timesteps=[0], 
                    output_deltas = False))
    
    w = WeatherTrainer(data,conf=config,model=model)
    w.preserved_conf.batch_size = 1
    w.preserved_conf.use_neoloader = False
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.log_every = 20
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 5000
    w.preserved_conf.lr_sched.warmup_end_step = 10
    w.preserved_conf.lr_sched.lr = 1e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.do_audit = True
    w.preserved_conf.yolo = False
    w.preserved_conf.restep_num = 50
    w.run()


@launch(ddp=1,nodes={'miramar':6})
def Dec19_ohp_l1era5():
    #config.gpus = '1'
    config.nope = 0
    config.prefix = "_adapt"
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,levels=levels_tiny,source='hres')
    dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
    dsc2 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc1,
                                           output_mesh = meshes.LatLonGrid(config=dsc2),
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2022, 1, 5),D(2023, 8, 25))),
                                           odds_idc = 1,
                                           use_mmap = True
                                           ))
    data.check_for_dates()
    data[0]
    #model = LinearAdapter(ForecastStepConfig(data.config.mesh, 
    #                output_mesh = data.config.output_mesh,
    #                timesteps=[0], 
    #                output_deltas = False))

    model = ForecastStepAdapter2(ForecastStepConfig(data.config.mesh, 
                                                output_mesh = data.config.output_mesh,
                                                patch_size=(4,8,8), 
                                                hidden_dim=768, 
                                                adapter_swin_depth=8,
                                                enc_swin_depth=0,
                                                dec_swin_depth=0, 
                                                proc_swin_depth=0, 
                                                timesteps=[0], 
                                                dims_per_head=32,
                                                use_matepoint = False,
                                                output_deltas = False))
    
    w = WeatherTrainer(data,conf=config,model=model)
    w.preserved_conf.batch_size = 1
    w.preserved_conf.use_neoloader = False
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.l2_loss = False
    w.preserved_conf.validate_every = -1
    w.preserved_conf.log_every = 20
    w.preserved_conf.save_every = 1000
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 20_000
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    w.preserved_conf.lr_sched.div_factor = 2 

    w.preserved_conf.lr_sched.lr = 3e-4 * 0.75
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.rerun_sample = 2
    w.preserved_conf.yolo = False
    w.preserved_conf.use_shampoo = 1
    w.run()

@launch(ddp=1,nodes={'miramar':6})
def Dec19_newenc():
    #config.gpus = '1'
    config.nope = 0
    config.prefix = "_adapt"
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,levels=levels_tiny,source='hres')
    dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
    dsc2 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc1,
                                           output_mesh = meshes.LatLonGrid(config=dsc2),
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2022, 1, 5),D(2023, 8, 25))),
                                           odds_idc = 1,
                                           use_mmap = True
                                           ))
    data.check_for_dates()
    data[0]
    #model = LinearAdapter(ForecastStepConfig(data.config.mesh, 
    #                output_mesh = data.config.output_mesh,
    #                timesteps=[0], 
    #                output_deltas = False))

    model = ForecastStepAdapter2(ForecastStepConfig(data.config.mesh, 
                                                output_mesh = data.config.output_mesh,
                                                patch_size=(4,8,8), 
                                                hidden_dim=768, 
                                                adapter_swin_depth=4,
                                                enc_swin_depth=0,
                                                dec_swin_depth=0, 
                                                proc_swin_depth=0, 
                                                timesteps=[0], 
                                                dims_per_head=32,
                                                use_matepoint = False,
                                                output_deltas = False))
    
    w = WeatherTrainer(data,conf=config,model=model)
    w.preserved_conf.batch_size = 1
    w.preserved_conf.use_neoloader = False
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.l2_loss = False
    w.preserved_conf.validate_every = -1
    w.preserved_conf.log_every = 20
    w.preserved_conf.save_every = 1000
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 20_000
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    w.preserved_conf.lr_sched.div_factor = 2 

    w.preserved_conf.lr_sched.lr = 3e-4
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.rerun_sample = 2
    w.preserved_conf.yolo = False
    w.preserved_conf.use_shampoo = 1
    w.run()


@launch(ddp=1,nodes={'martins':1},start_method='spawn')
#@launch(ddp=1,nodes={'bimini':6},start_method='spawn')
#@launch(ddp=1,nodes={'halfmoon':4})
#@launch(ddp=0)
def Dec23_modeldev():
    #config.gpus = '0'
    config.nope = 1
    config.prefix = "_adapt"
    config.name+= "hres-gfs-rambuffer-l2"
    #config.resume = "_"+config.activity.replace("_","-")+"_"


    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='gfs-28')
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1, mesh2],
                                           outputs = [mesh3],
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2020, 1, 1),D(2023, 8, 25))),
                                           ))

    data.check_for_dates()
    if 0:
        sample = data[0]
        x = sample[0][0]
        y = sample[1][0]
        imshow_compare(x,data.config.mesh,y,data.config.output_mesh)
        exit()

    import model_unet
 
    model_conf = ForecastStepConfig(data.config.inputs, 
                outputs = data.config.outputs,
                patch_size=(4,8,8), 
                hidden_dim=768, 
                enc_swin_depth=0,
                dec_swin_depth=0, 
                proc_swin_depth=0, 
                adapter_swin_depth=8,
                timesteps=[0], 
                output_deltas = True,
                use_matepoint = False )
    
    model = model_unet.ForecastStepAdapterConv(model_conf)
    # print number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1e6:0.1f}M")
    if 0:
        model.half().to('cuda')
        sample = default_collate([data[0]])
        x = sample[0][0].to('cuda')
        timer = Timer(print=True,torch_sync=True); mem = GPUMemoryMonitor(print=True,torch_sync=True)
        with timer, mem:
            y = model(x)
        exit()
    
    w = WeatherTrainer(data,conf=config,model=model)
    w.preserved_conf.batch_size = 1
    w.preserved_conf.use_neoloader = True
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.l2_loss = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.log_every = 20
    w.preserved_conf.save_every = 1000
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 30_000
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.div_factor = 2 
    w.preserved_conf.lr_sched.lr = 3.5e-4
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.rerun_sample = 1
    w.preserved_conf.yolo = False
    w.preserved_conf.use_shampoo = 1
    w.run()


#@launch(ddp=1,nodes={'stinson':4})
@launch(ddp=0)
def Dec23_debug():
    #config.gpus = '0'
    config.nope = 0
    config.prefix = "_adapt"
    config.name+= "simple_gfs_deltas"
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,levels=levels_tiny,source='hres')
    dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='gfs')
    dsc2 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc1,
                                           output_mesh = meshes.LatLonGrid(config=dsc2),
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2022, 1, 5),D(2023, 8, 25))),
                                           #requested_dates = get_dates((D(2022, 1, 5),D(2022, 1, 10))),
                                           odds_idc = 1,
                                           #use_mmap = True
                                           ))
    data.check_for_dates()
    #data[0]
    import model_unet
    model_conf = ForecastStepConfig(data.config.mesh, 
                output_mesh = data.config.output_mesh,
                patch_size=(4,8,8), 
                hidden_dim=768, 
                enc_swin_depth=0,
                dec_swin_depth=0, 
                proc_swin_depth=0, 
                adapter_swin_depth=8,
                timesteps=[0], 
                output_deltas = True,
                use_matepoint = False )
    
    model = model_unet.ForecastStepAdapterConv2(model_conf)
    # print number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1e6:0.1f}M")
    if 1:
        model.half().to('cuda')
        Nlev = 28 ; Npr = 5; Nsfc = 6 ; N = Nlev*Npr+Nsfc
        x = torch.cat((torch.arange(0,Nlev*Npr).reshape(Nlev,Npr).permute(1,0).flatten(),torch.arange(Nlev*Npr,N)))
        x = x.expand(1,720,1440,N).float().half().to('cuda')
        timer = Timer(print=True,torch_sync=True); mem = GPUMemoryMonitor(print=True,torch_sync=True)
        with timer, mem:
            y = model(x)
        exit()


@launch(ddp=False,nodes={'martins':1})
def Dec12_GFS_baseline():
    config.gpus = '0'
    config.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(2000, 1, 23),D(2017, 12,30))),
                                           only_at_z=[0,12],
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32, only_at_z=[0,12]))
    model.output_deltas = True
    model.do_sub = False
    
    load_path = '/fast/ignored/runs/run_Dec4-tardis-l2ft_20231205-234837/model_epoch19_iter249588_loss0.014.pt'
    checkpoint = torch.load(load_path,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)


    w = WeatherTrainer(data,conf=config,model=model)

    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    w.preserved_conf.l2_loss = True
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = False
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 250_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.025e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(2000, 1, 23),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))

    w.run()


@launch(ddp=False)
def datacompare():
        #config.gpus = '0'
    config.nope = 0
    config.prefix = "_adapt"
    config.name+= "simple_gfs_deltas"
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres')
    #dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
    dsc2 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc1,
                                           output_mesh = meshes.LatLonGrid(config=dsc2),
                                           timesteps=[0],
                                           #requested_dates = get_dates((D(2022, 1, 5),D(2023, 8, 25))),
                                           requested_dates = get_dates((D(2022, 1, 5),D(2022, 1, 10))),
                                           odds_idc = 1,
                                           #use_mmap = True
                                           ))
    data.check_for_dates()
    dat = default_collate([data[3]])
    x = dat[0][0].to('cuda')[:,:,:,:-2]
    #x = unnorm(x,data.config.mesh)
    x = interp_levels(x,data.config.mesh,data.config.mesh.levels,data.config.output_mesh.levels)
    x = unnorm(x,data.config.output_mesh)
    y = unnorm(dat[1][0].to('cuda'),data.config.output_mesh)
    from eval import compute_errors
    with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float32):
        rms = compute_errors(x,y,data.config.output_mesh,doprint=True)

    data[0]

@launch(ddp=False,nodes={'martins':1})
def Dec12_GFS():
    config.gpus = '0'
    #config.nope = True
    config.name = config.name + "ohp"
    dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    dsc2 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='gfs')
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc2,
                                           output_mesh = meshes.LatLonGrid(config=dsc1),
                                           timesteps=[0,24],
                                           requested_dates = get_dates((D(2022, 1, 5),D(2023, 8, 25))),
                                           only_at_z=[0,12],
                                           odds_idc = 0.2,
                                           use_mmap = True
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.output_mesh, adapter_swin_depth=4, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32, only_at_z=[0,12]))
    load_path = '/fast/ignored/runs/run_Dec4-tardis-l2ft_20231205-234837/model_epoch19_iter249588_loss0.014.pt'
    checkpoint = torch.load(load_path,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    adapter = ForecastStepAdapter(model)
    w = AdapterTrainer(data,model,adapter,config=config)
    w.run()

@launch(ddp=False,nodes={'martins':1})
def Dec12_GFS2():
    config.gpus = '1'
    config.name = config.name + "ohp4swin"
    dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    dsc2 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='gfs')
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc2,
                                           output_mesh = meshes.LatLonGrid(config=dsc1),
                                           timesteps=[0,24],
                                           requested_dates = get_dates((D(2022, 1, 5),D(2023, 8, 25))),
                                           only_at_z=[0,12],
                                           odds_idc = 0.2,
                                           use_mmap = True
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.output_mesh, adapter_swin_depth=4, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32, only_at_z=[0,12]))
    load_path = '/fast/ignored/runs/run_Dec4-tardis-l2ft_20231205-234837/model_epoch19_iter249588_loss0.014.pt'
    checkpoint = torch.load(load_path,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    adapter = ForecastStepAdapter(model)
    w = AdapterTrainer(data,model,adapter,config=config)
    w.run()


@launch(ddp=0,start_method='spawn')
def Dec25_Mario():
    #config.gpus = '0'
    config.nope = 1
    config.prefix = "_adapt"
    config.name+= "hres-gfs-2rerun-l2"
    #config.resume = "_"+config.activity.replace("_","-")+"_"


    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='gfs-28')
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1, mesh2],
                                           outputs = [mesh3],
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2020, 1, 1),D(2023, 8, 25))),
                                           ))

    model_conf = ForecastStepConfig(data.config.inputs, 
                outputs = data.config.outputs,
                patch_size=(4,8,8), 
                hidden_dim=768, 
                enc_swin_depth=0,
                dec_swin_depth=0, 
                proc_swin_depth=0, 
                adapter_swin_depth=8,
                timesteps=[0], 
                output_deltas = True,
                use_matepoint = False )
    import model_unet
    model = model_unet.ForecastStepAdapterConv(model_conf)
    # print number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1e6:0.1f}M")
    
    w = WeatherTrainer(data,conf=config,model=model)
    w.preserved_conf.batch_size = 1
    w.preserved_conf.use_neoloader = True
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.l2_loss = True
    w.preserved_conf.validate_every = 2
    w.preserved_conf.log_every = 20
    w.preserved_conf.save_every = 1000
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 30_000
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.div_factor = 2 
    w.preserved_conf.lr_sched.lr = 3.5e-4
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.rerun_sample = 1
    w.preserved_conf.yolo = False
    w.preserved_conf.use_shampoo = 0
    w.run()


@launch(ddp=1,nodes={'stinson':4},port=29500, start_method="spawn")
def Dec22_neoenc_ft():
    config.gpus = '0'
    config.nope = True
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[24], max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           only_at_z=[0,12],
                                           requested_dates = get_dates((D(1997, 1, 23),D(2017, 12, 28))),
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12,
                                                  lat_compress=False, timesteps=[24], dims_per_head=32, processor_dt=12,use_matepoint=False))
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = am_i_torchrun()
    w.preserved_conf.initial_gradscale = 65536.0 * 8
    w.preserved_conf.l2_loss = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 18_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.03e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    w.preserved_conf.dates = get_dates((D(1997, 1, 23),D(2017, 12,28)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 28)))
    w.run()

if __name__ == '__main__':
    run(locals().values())
