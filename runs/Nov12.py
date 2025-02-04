from launch import * 
from train import *
# from train_adapter import *
from hres.model import HresModel
from torch.utils import checkpoint as torch_checkpoint

@launch(nodes={'glass': 6},port=29502, clear_cache=False, start_method="spawn")#, zulip=1, ping='@**Joan Creus-Costa**')
def Nov2_doctor_neohresonly():
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.resume = "/huge/deep/runs/model_epoch1_iter138392_step17299_loss0.034.pt"
    #config.resume = "/huge/deep/runs/model_epoch1_iter139992_step17499_loss0.032.pt"
    config.nope = False
    tdates = get_dates([(D(1979, 5, 1), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 2, 21))])

    extra_input = ['zeropad', '45_tcc', '034_sstk', '168_2d', 'zeropad', 'zeropad', 'zeropad', 'zeropad',
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_input = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    #imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_input, input_levels=levels_medium, levels=levels_joank)
    #omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

    #mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_pad=len(extra_input), input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(source='neohres-20', extra_sfc_vars=extra_input, input_levels=levels_hres,levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

    timesteps = [6,12,24,30,48,96,120] #[6, 12, 24, 72, 144]
    #timesteps = [6,24,30,48,96]#,120]
    timesteps = [6,12,24,30,48,96,120]
    timesteps = [6,12,24,30]#,96,120]
    timesteps = [1,2,3,6]#,12,24]
    timesteps = list(range(25))
    #timesteps = [6,12,24,30]#,96,120]
    timesteps = [0,6,12,18,24,30,48,96]
    timesteps = [0,6,18,24]
    config.loss_consts = {6: 1.0, 12: 1.0, 24: 1.4, 30: 0.7, 48: 0.4, 96: 0.2, 120: 0.1,  } # <-------- new weights
    config.loss_consts_override = {24: 1.3}
    data = WeatherDataset(DataConfig(inputs=[mesh2], outputs=[omesh],
                                           timesteps=timesteps,
                                           #only_at_z = list(range(24)),
                                           #random_timestep_subset = 3,
                                           requested_dates = tdates
                                           ))

    #import evals.package_neo as pn
    #model = pn.get_ducatidoctorate()
    #modelw = pn.get_latentdoctorate()
    #model.config.checkpointfn = matepoint.checkpoint
    #model.config.matepoint_table = 'auto'
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh2],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,8,8),
            hidden_dim=1536, #1408,
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            #timesteps=timesteps,
            #checkpoint_convs=True,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            #parallel_encoders=True,
            neorad=True,
            #matepoint_table = 'auto',
        window_size=(3,5,7)))
    #model.load_state_dict(modelw.state_dict(),strict=True) 

    """
    from pprint import pprint
    print("model")
    pprint([x[0] for x in model.state_dict().items()])

    print("modelw")
    pprint([x[0] for x in modelw.state_dict().items()])
    exit()
    """

    """
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
    """

    #del modelw
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.encoder.parameters():
        param.requires_grad = True

    config.HALF = True
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.validate_every = -1
    #config.compute_Bcrit_every = 1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.save_optimizer = True
    config.save_imgs = False
    config.optim = 'shampoo'
    #config.optim = 'adam'
    #config.steamroll_over_mismatched_dims = "Forgive me father for I have sinned doctorate"
    config.shampoo.dim = 4096
    config.shampoo.version = 'old'
    #config.optim = 'adam'
    config.reset_optimizer = True
    config.reset_steps_on_resume = False#True
    #config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    # config.lr_sched.cooosine_period = 25_000 #
    config.lr_sched.cosine_period = 8_000 # <------- new
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 500
    #config.lr_sched.restart_warmup_end_step = 0 # <------- so that we don't do a nested warmup to the warmup rate
    config.lr_sched.lr = 0.05e-3
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    #config.initial_gradscale = 1024.
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



#@launch(ddp=0,start_method='spawn')
@launch(nodes={'muir': 6},port=29502, clear_cache=False, start_method="spawn")#, zulip=1, ping='@**Joan Creus-Costa**')
def Nov2_doctor_neohresgfs_blend():
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.resume = "/huge/deep/runs/model_epoch1_iter138392_step17299_loss0.034.pt"
    #config.resume = "/huge/deep/runs/model_epoch1_iter139992_step17499_loss0.032.pt"
    config.nope = False
    tdates = get_dates([(D(1979, 5, 1), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])

    extra_input = ['zeropad', '45_tcc', '034_sstk', '168_2d', 'zeropad', 'zeropad', 'zeropad', 'zeropad',
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_input = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    #imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_input, input_levels=levels_medium, levels=levels_joank)
    #omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

    mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_vars=extra_input, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(source='neohres-20', extra_sfc_vars=extra_input, input_levels=levels_hres,levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

    timesteps = [6,12,24,30,48,96,120] #[6, 12, 24, 72, 144]
    #timesteps = [6,24,30,48,96]#,120]
    timesteps = [6,12,24,30,48,96,120]
    timesteps = [6,12,24,30]#,96,120]
    timesteps = [1,2,3,6]#,12,24]
    timesteps = list(range(25))
    #timesteps = [6,12,24,30]#,96,120]
    timesteps = [0,6,12,18,24,30,48,96]
    timesteps = [0,6,18,24]
    config.loss_consts = {6: 1.0, 12: 1.0, 24: 1.4, 30: 0.7, 48: 0.4, 96: 0.2, 120: 0.1,  } # <-------- new weights
    config.loss_consts_override = {24: 1.3}
    data = WeatherDataset(DataConfig(inputs=[mesh1, mesh2], outputs=[omesh],
                                           timesteps=timesteps,
                                           #only_at_z = list(range(24)),
                                           #random_timestep_subset = 3,
                                           requested_dates = tdates
                                           ))

    import evals.package_neo as pn
    #model = pn.get_ducatidoctorate()
    modelw = pn.get_latentdoctorate()
    #model.config.checkpointfn = matepoint.checkpoint
    #model.config.matepoint_table = 'auto'
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,8,8),
            hidden_dim=1536, #1408,
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            #timesteps=timesteps,
            #checkpoint_convs=True,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            parallel_encoders=True,
            neorad=True,
            #matepoint_table = 'auto',
        window_size=(3,5,7)))

    """
    from pprint import pprint
    print("model")
    pprint([x[0] for x in model.state_dict().items()])

    print("modelw")
    pprint([x[0] for x in modelw.state_dict().items()])
    exit()
    """

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

    del modelw
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.encoders.parameters():
        param.requires_grad = True

    config.HALF = True
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.validate_every = -1
    #config.compute_Bcrit_every = 1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.save_optimizer = True
    config.save_imgs = False
    config.optim = 'shampoo'
    #config.optim = 'adam'
    #config.steamroll_over_mismatched_dims = "Forgive me father for I have sinned doctorate"
    config.shampoo.dim = 2048
    config.shampoo.version = 'old'
    #config.optim = 'adam'
    config.reset_optimizer = False#True
    config.reset_steps_on_resume = False#True
    #config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    # config.lr_sched.cooosine_period = 25_000 #
    config.lr_sched.cosine_period = 8_000 # <------- new
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 500
    #config.lr_sched.restart_warmup_end_step = 0 # <------- so that we don't do a nested warmup to the warmup rate
    config.lr_sched.lr = 0.05e-3
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    #config.initial_gradscale = 1024.
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'muir': 6},port=29503, clear_cache=False, start_method="spawn", zulip=1, ping='@**Joan Creus-Costa**')
def Nov1_bigducati():
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.resume = "/huge/deep/runs/model_epoch1_iter138392_step17299_loss0.034.pt"
    #config.resume = "/huge/deep/runs/model_epoch1_iter139992_step17499_loss0.032.pt"
    config.nope = False
    tdates = get_dates([(D(1979, 5, 1), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])

    extra_input = ['zeropad', '45_tcc', '034_sstk', '168_2d', 'zeropad', 'zeropad', 'zeropad', 'zeropad',
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_input = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_input, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,12,24,30,48,96,120] #[6, 12, 24, 72, 144]
    #timesteps = [6,24,30,48,96]#,120]
    timesteps = [6,12,24,30,48,96,120]
    timesteps = [6,12,24,30]#,96,120]
    timesteps = [1,2,3,6]#,12,24]
    #timesteps = [6,12,24,30]#,96,120]
    timesteps = [0,6,12,18,24,30,48,96]
    timesteps = list(range(25))
    timesteps = [1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,19,20,21,22,23]
    config.loss_consts = {6: 1.0, 12: 1.0, 24: 1.4, 30: 0.7, 48: 0.4, 96: 0.2, 120: 0.1,  } # <-------- new weights
    config.loss_consts_override = {24: 1.3}
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
                                           only_at_z = list(range(24)),
                                           random_timestep_subset = 4,
                                           requested_dates = tdates
                                           ))

    import evals.package_neo as pn
    model = pn.get_bigducatidoctorate()
    #model = pn.get_latentdoctorate()
    model.config.checkpointfn = matepoint.checkpoint

    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.processors['1'].parameters():
        param.requires_grad = True
    #model.config.matepoint_table = 'auto'
    """
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
            #timesteps=timesteps,
            #checkpoint_convs=True,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
            matepoint_table = 'auto',
        window_size=(3,5,7)))
    """

    config.HALF = True
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.validate_every = -1
    #config.compute_Bcrit_every = 1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.save_optimizer = True
    config.save_imgs = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    #config.steamroll_over_mismatched_dims = "Forgive me father for I have sinned doctorate"
    config.shampoo.dim = 8192
    config.shampoo.version = 'old'
    #config.optim = 'adam'
    config.reset_optimizer = True#True
    config.reset_steps_on_resume = False#True
    #config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    # config.lr_sched.cooosine_period = 25_000 #
    config.lr_sched.cosine_period = 20_000 # <------- new
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 500
    #config.lr_sched.restart_warmup_end_step = 0 # <------- so that we don't do a nested warmup to the warmup rate
    config.lr_sched.lr = 0.1e-3
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    #config.initial_gradscale = 1024.
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'bimini': 6},port=29503, clear_cache=False, start_method="spawn", zulip=1, ping='@**Joan Creus-Costa**')
def Nov1_ducati():
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.resume = "/huge/deep/runs/model_epoch1_iter138392_step17299_loss0.034.pt"
    #config.resume = "/huge/deep/runs/model_epoch1_iter139992_step17499_loss0.032.pt"
    config.nope = False
    tdates = get_dates([(D(1979, 5, 1), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])

    extra_input = ['zeropad', '45_tcc', '034_sstk', '168_2d', 'zeropad', 'zeropad', 'zeropad', 'zeropad',
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_input = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp', '143_cp', '201_mx2t', '202_mn2t', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_input, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,12,24,30,48,96,120] #[6, 12, 24, 72, 144]
    #timesteps = [6,24,30,48,96]#,120]
    timesteps = [6,12,24,30,48,96,120]
    timesteps = [6,12,24,30]#,96,120]
    timesteps = [1,2,3,6]#,12,24]
    #timesteps = [6,12,24,30]#,96,120]
    timesteps = [0,6,12,18,24,30,48,96]
    timesteps = list(range(25))
    timesteps = [1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,19,20,21,22,23]
    config.loss_consts = {6: 1.0, 12: 1.0, 24: 1.4, 30: 0.7, 48: 0.4, 96: 0.2, 120: 0.1,  } # <-------- new weights
    config.loss_consts_override = {24: 1.3}
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
                                           only_at_z = list(range(0, 24, 6)),
                                           random_timestep_subset = 4,
                                           requested_dates = tdates
                                           ))

    import evals.package_neo as pn
    model = pn.get_ducatidoctorate()
    #model = pn.get_latentdoctorate()
    model.config.checkpointfn = matepoint.checkpoint

    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.processors['1'].parameters():
        param.requires_grad = True
    #model.config.matepoint_table = 'auto'
    """
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
            #timesteps=timesteps,
            #checkpoint_convs=True,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
            matepoint_table = 'auto',
        window_size=(3,5,7)))
    """

    config.HALF = True
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.validate_every = -1
    #config.compute_Bcrit_every = 1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.save_optimizer = True
    config.save_imgs = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    #config.steamroll_over_mismatched_dims = "Forgive me father for I have sinned doctorate"
    config.shampoo.dim = 8192
    config.shampoo.version = 'old'
    #config.optim = 'adam'
    config.reset_optimizer = True#True
    config.reset_steps_on_resume = False#True
    #config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    # config.lr_sched.cooosine_period = 25_000 #
    config.lr_sched.cosine_period = 20_000 # <------- new
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 500
    #config.lr_sched.restart_warmup_end_step = 0 # <------- so that we don't do a nested warmup to the warmup rate
    config.lr_sched.lr = 0.035e-3
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    #config.initial_gradscale = 1024.
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'muir.fast': 6, 'bimini': 6},port=29502, clear_cache=False, start_method="spawn")#, zulip=1, ping='@**Joan Creus-Costa**')
def Oct29_doctorate_ft():
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.resume = "/huge/deep/runs/model_epoch1_iter138392_step17299_loss0.034.pt"
    #config.resume = "/huge/deep/runs/model_epoch1_iter139992_step17499_loss0.032.pt"
    config.nope = False
    tdates = get_dates([(D(1979, 5, 1), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])

    extra_input = ['zeropad', '45_tcc', '034_sstk', '168_2d', 'zeropad', 'zeropad', 'zeropad', 'zeropad',
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_input = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_input, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,12,24,30,48,96,120] #[6, 12, 24, 72, 144]
    #timesteps = [6,24,30,48,96]#,120]
    timesteps = [6,12,24,30,48,96,120]
    timesteps = [6,12,24,30]#,96,120]
    timesteps = [1,2,3,6]#,12,24]
    timesteps = list(range(25))
    #timesteps = [6,12,24,30]#,96,120]
    timesteps = [0,6,12,18,24,30,48,96]
    config.loss_consts = {6: 1.0, 12: 1.0, 24: 1.4, 30: 0.7, 48: 0.4, 96: 0.2, 120: 0.1,  } # <-------- new weights
    config.loss_consts_override = {24: 1.3}
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
                                           #only_at_z = list(range(24)),
                                           random_timestep_subset = 3,
                                           requested_dates = tdates
                                           ))

    #import evals.package_neo as pn
    #model = pn.get_ducatidoctorate()
    #model = pn.get_latentdoctorate()
    #model.config.checkpointfn = matepoint.checkpoint
    #model.config.matepoint_table = 'auto'
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
            #timesteps=timesteps,
            #checkpoint_convs=True,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
            matepoint_table = 'auto',
        window_size=(3,5,7)))

    config.HALF = True
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.validate_every = -1
    #config.compute_Bcrit_every = 1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.save_optimizer = True
    config.save_imgs = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    #config.steamroll_over_mismatched_dims = "Forgive me father for I have sinned doctorate"
    config.shampoo.dim = 2048
    config.shampoo.version = 'old'
    #config.optim = 'adam'
    config.reset_optimizer = True#True
    config.reset_steps_on_resume = False#True
    #config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    # config.lr_sched.cooosine_period = 25_000 #
    config.lr_sched.cosine_period = 4_000 # <------- new
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 500
    #config.lr_sched.restart_warmup_end_step = 0 # <------- so that we don't do a nested warmup to the warmup rate
    config.lr_sched.lr = 0.035e-3
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    #config.initial_gradscale = 1024.
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'ip-172-31-13-240': 8},port=29500, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')
def Sep4_cloudmaster():
    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1975, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    #omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25', extra_sfc_vars=[], extra_sfc_pad=len(extra), input_levels=levels_joank, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25', extra_sfc_vars=extra, input_levels=levels_joank, levels=levels_joank)
    timesteps = [6, 12, 24, 72, 144]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(70e9),
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
            timesteps=timesteps,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
        window_size=(3,5,7)))

    config.HALF = True
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.shampoo.version = 'new'
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 15_000
    config.lr_sched.cosine_period = 25_000 # <------- new
    config.lr_sched.step_offset = -10_899 # <---------new, set to #steps model is at
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 200
    config.lr_sched.restart_warmup_end_step = 0 # <------- new, so that we don't do a nested warmup to the warmup rate
    # Scaling by sqrt(3) now that we're using 
    config.lr_sched.lr = 0.04e-3 
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.loss_consts = {144: 0.075, 72: 0.3, 24: 1.2, 0: 0.2, 6: 1.0}
    config.loss_consts = {144: 0.1, 72: 0.33, 24: 1.3, 0: 0.5, 6: 1.0, 12: 1.0} # <-------- new weights
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'bimini': 5},port=29504, start_method="spawn")#, zulip=0, ping='@**Joan Creus-Costa**')
def Sep10_relhum():
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = False
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    #omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=[], extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           use_rh=True
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,16,16), 
            hidden_dim=1024, 
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
    config.ignore_train_safegaurd = False
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 500
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 50_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1500
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    # Scaling by sqrt(3) now that we're using 
    config.lr_sched.lr = 0.3e-3 
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.loss_consts = {48: 0.1, 24: 1.0, 6: 1.0, 3: 1.0}
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'bimini': 6},port=29504, start_method="spawn")#, zulip=True, ping='@**Anuj Shetty**')
def Sep8_ramtesting():
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    # tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 5, 1))])
    extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra_all, output_only_vars=extra_out_only, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra_all, output_only_vars=extra_out_only, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24, 36]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
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
            hidden_dim=1408, #1408, 
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
    config.optim = 'adam'
    config.reset_optimizer = True
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




@launch(nodes={'muir': 6},port=29505, start_method="spawn")#, zulip=True, ping='@**John Dean**',validate=False,kill_nvidia=True)
def Sep4_johnsblundering():
    #config.gpus = '0-2'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = False
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_vars=extra, output_only_vars=extra, is_output_only=True, input_levels=levels_gfs, levels=levels_joank)
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_pad=3, is_output_only=True, input_levels=levels_gfs, levels=levels_joank)
    #mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='hres-13', extra_sfc_pad=3, is_output_only=True,  input_levels=levels_tiny,levels=levels_joank)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
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
            hidden_dim=512,
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            timesteps=timesteps,
            dims_per_head=32,
            processor_dt=6,
            neorad=True,
        window_size=(3,5,7)))

    yield model
    config.loss_consts = {24: 1.0, 24*6: 1, 24*12: 0.5, 24*24: 0.25}
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
    config.lr_sched.cosine_period = 15000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 300
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 3e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'barceloneta': 6, 'miramar.fast': 6},port=29501, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')
def Sep7_doctorate():
    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    config.disregard_buffer_checksum = False#True
    #config.nope = True
    # tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 4, 30))])
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
    config.lr_sched.restart_warmup_end_step = 500
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



@launch(nodes={'bimini': 6},port=29504, start_method="spawn", zulip=1, ping='@**Joan Creus-Costa**')
def Sep2_apkid():
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24, 48]
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
            patch_size=(5,16,16), 
            hidden_dim=1024, 
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
    config.ignore_train_safegaurd = False
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 500
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 50_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1500
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    # Scaling by sqrt(3) now that we're using 
    config.lr_sched.lr = 0.3e-3 
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.loss_consts = {48: 0.1, 24: 1.0, 6: 1.0, 3: 1.0}
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'bimini': 6},port=29504, start_method="spawn", zulip=0, ping='@**Joan Creus-Costa**')
def Sep1_middleschooler():
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = False
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
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
            patch_size=(5,30,30), 
            hidden_dim=1152, 
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
    config.ignore_train_safegaurd = False
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 500
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 50_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1500
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    # Scaling by sqrt(3) now that we're using 
    config.lr_sched.lr = 0.3e-3 
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.loss_consts = {48: 0.1, 24: 1.0, 6: 1.0, 3: 1.0}
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'muir': 6},port=29504, start_method="spawn", zulip=0, ping='@**Joan Creus-Costa**')
def Aug30_highschooler():
    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24]
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
            patch_size=(5,16,16), 
            hidden_dim=1024, 
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
    config.ignore_train_safegaurd = False
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 500
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 50_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1500
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    # Scaling by sqrt(3) now that we're using 
    config.lr_sched.lr = 0.3e-3 
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.loss_consts = {48: 0.1, 24: 1.0, 6: 1.0, 3: 1.0}
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'barceloneta': 6, 'miramar.fast': 6},port=29504, start_method="spawn", zulip=True, ping='@**Joan Creus-Costa**')
def Aug30_neomaster48():
    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24, 48]
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
    config.shampoo.dim = 8192
    #config.optim = 'adam'
    config.reset_optimizer = True
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 15_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1500
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    # Scaling by sqrt(3) now that we're using 
    config.lr_sched.lr = 0.05e-3 
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.loss_consts = {48: 0.3, 24: 1.0, 6: 1.0, 3: 1.0}
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'baga': 6},port=29504, start_method="spawn")
def Aug14_hwtest():
    import natten
    #assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

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
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=3, 
            output_deltas=False, 
            decoder_reinput_initial=False, 
            # decoder_reinput_size=96, 
            neorad=True, 
        window_size=(3,5,7)))

    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 31_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    # Scaling by sqrt(3) now that we're using 
    config.lr_sched.lr = 0.3e-3  * (0.22499999999999998 / 0.1575973753258069)
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    #config.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'barceloneta': 6, 'miramar.fast': 6},port=29504, start_method="spawn", zulip=True, ping='@**Joan Creus-Costa**')
def Aug19_master48ft():
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
    config.loss_consts = {48: 0.4, 24: 1.0, 6: 1.0, 3: 1.0}
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'stinson': 4, 'singing.fast': 4},port=29504, start_method="spawn", zulip=True, ping='@**Haoxing Du**',validate=True)
def Aug19_bachelorette():
    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    # config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24]
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
            patch_size=(5,6,6), 
            hidden_dim=768, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=6, 
            output_deltas=False, 
            decoder_reinput_initial=False,
            neorad=True, 
        window_size=(3,5,7)))

    yield model

    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 20
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
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
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'stinson': 4, 'singing.fast': 4},port=29504, start_method="spawn", zulip=True, ping='@**Haoxing Du**')
def Aug14_master624():
    import natten
    #assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 24]
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
            processor_dt=6, 
            use_matepoint=True, 
            output_deltas=False, 
            decoder_reinput_initial=False, 
            # decoder_reinput_size=96, 
            neorad=True, 
        window_size=(3,5,7)))

    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 20
    config.optim = 'shampoo'
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
    config.lr_sched.lr = 0.2e-3 
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    #config.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


#@launch(ddp=0,start_method='spawn')
#@launch(nodes={'barceloneta': 6},port=29504, start_method="spawn")
@launch(nodes={'muir': 6},port=29504, start_method="spawn")
def Aug14_master24fast():
    import natten
    #assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [3, 6, 24]
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
            # decoder_reinput_size=96, 
            neorad=True, 
        window_size=(3,5,7)))

    config.HALF = True
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 31_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    # Scaling by sqrt(3) now that we're using 
    config.lr_sched.lr = 0.3e-3  * (0.22499999999999998 / 0.1575973753258069)
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    #config.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'barceloneta': 6},port=29504, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Aug11_serpentbachelor():
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
    timesteps = [6,12,24,48,72]
    timesteps = [6,24]
    train_timesteps = [6, 24, 72]
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
    w.preserved_conf.lr_sched.cosine_period = 20_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    # Scaling by sqrt(3) now that we're using 
    w.preserved_conf.lr_sched.lr = 0.1e-3 
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'barceloneta': 6},port=29504, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Jul29_bachelor():
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
    timesteps = [6,12,24,48,72]
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8), 
                                                  hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
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
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    # Scaling by sqrt(3) now that we're using 
    w.preserved_conf.lr_sched.lr = 0.3e-3 
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'miramar': 6},start_method='spawn')
def Jun3_deltas_perturb():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = False
    config.coupled.hell = True
    config.coupled.B = 256
    config.coupled.freeze = False
    config.coupled.config.weights = np.array([3, 3, 3, 4, 4, 0.1, 0.1], dtype=np.float32)
    config.coupled.config.weight = 0.5
    config.coupled.model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=512, absbias=True, depth=16, silly_rel=True, pressure_vertical=5, sfc_extra=3)
    #check = torch.load('/fast/model_step417500_loss0.140.pt', map_location='cpu')
    #config.coupled.model.load_state_dict(check['model_state_dict'],strict=True)
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)

    #timesteps = [6,24]
    timesteps = [6,24,72]
    timesteps = [24,72]
    timesteps = [6, 12]
    timesteps = [6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(5,4,4), hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=False, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(2,3,6)))

    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(5,4,4), hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=4, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=True, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(2,3,6), perturber=0.5))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,4,4), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(4,3,6)))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 250
    w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 2000
    w.preserved_conf.lr_sched.lr = (1/np.e)*1e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.HALF = True
    w.preserved_conf.val_dates = vdates
    w.preserved_conf.initial_gradscale = 16384.
    w.run()



@launch(nodes={'singing.fast': 5, 'miramar.fast': 5, 'bimini': 5},start_method='spawn')
def Jun3_neodeltas():
    config.gpus = '0-5' #0,1,2,3,5
    config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = False
    config.coupled.hell = True
    config.coupled.B = 256
    config.coupled.freeze = False
    config.coupled.config.weights = np.array([3, 3, 3, 4, 4, 0.1, 0.1], dtype=np.float32)
    config.coupled.config.weight = 0.75
    config.coupled.model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=512, absbias=True, depth=16, silly_rel=True, pressure_vertical=5, sfc_extra=3)
    #check = torch.load('/fast/model_step417500_loss0.140.pt', map_location='cpu')
    #config.coupled.model.load_state_dict(check['model_state_dict'],strict=True)
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)

    #timesteps = [6,24]
    timesteps = [6,24,72]
    timesteps = [24,72]
    timesteps = [6, 12]
    timesteps = [6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(5,4,4), hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=False, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(2,3,6)))

    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(5,4,4), hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=4, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=True, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(2,3,6), perturber=0))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,4,4), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(4,3,6)))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 250
    w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 2000
    w.preserved_conf.lr_sched.lr = (1/np.e)*1e-3 * 0.8
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.HALF = True
    w.preserved_conf.val_dates = vdates
    w.preserved_conf.initial_gradscale = 16384.
    w.run()


@launch(nodes={'singing.fast': 4, 'stinson': 4},start_method='spawn')
def May31_perturbator():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = False
    config.coupled.hell = True
    config.coupled.B = 256
    config.coupled.freeze = False
    config.coupled.config.weights = np.array([3, 3, 3, 4, 4, 0.1, 0.1], dtype=np.float32)
    config.coupled.config.weight = 0.5
    config.coupled.model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=512, absbias=True, depth=16, silly_rel=True, pressure_vertical=5, sfc_extra=3)
    #check = torch.load('/fast/model_step417500_loss0.140.pt', map_location='cpu')
    #config.coupled.model.load_state_dict(check['model_state_dict'],strict=True)
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)

    #timesteps = [6,24]
    timesteps = [6,24,72]
    timesteps = [24,72]
    timesteps = [6, 12]
    timesteps = [3, 6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(5,4,4), hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=False, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(2,3,6)))

    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(5,8,8), hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=True, output_deltas=False, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(2,3,6), perturber=0.01))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,4,4), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(4,3,6)))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 250
    w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 2000
    w.preserved_conf.lr_sched.lr = (1/np.e)*1e-3 * 0.5
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.HALF = False
    w.preserved_conf.val_dates = vdates
    w.preserved_conf.initial_gradscale = 16384.
    w.run()


@launch(nodes={'miramar': 6, 'bimini.fast': 6},start_method='spawn')
def May23_hrestest():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = False
    config.coupled.hell = True
    config.coupled.B = 256
    config.coupled.freeze = False
    config.coupled.config.weights = np.array([3, 3, 3, 4, 4, 0.1, 0.1], dtype=np.float32)
    config.coupled.config.weight = 0.5
    config.coupled.model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=512, absbias=True, depth=16, silly_rel=True, pressure_vertical=5, sfc_extra=3)
    #check = torch.load('/fast/model_step417500_loss0.140.pt', map_location='cpu')
    #config.coupled.model.load_state_dict(check['model_state_dict'],strict=True)
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)

    #timesteps = [6,24]
    timesteps = [6,24,72]
    timesteps = [24,72]
    timesteps = [6, 12]
    timesteps = [3, 6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(5,4,4), hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=False, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(2,3,6)))

    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(5,8,8), hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=True, output_deltas=False, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(2,3,6)))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,4,4), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=128, neorad=True, window_size=(4,3,6)))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 250
    w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 2000
    w.preserved_conf.lr_sched.lr = (1/np.e)*1e-3 * 0.5
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.HALF = False
    w.preserved_conf.val_dates = vdates
    w.preserved_conf.initial_gradscale = 16384.
    w.run()




@launch(nodes={'ip-172-31-59-127':6},start_method='spawn', port=29111)
def May9_mccloudface():
    config.gpus = '0-7' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    config.loss_consts[24] = 1
    config.loss_consts[48] = 0.25
    config.coupled.hell = False
    config.coupled.B = 768
    config.coupled.config.weights = np.array([3, 3, 3, 4, 4, 0.1, 0.1], dtype=np.float32)
    config.coupled.config.weight = 0.5
    """
    config.coupled.model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True)
    check = torch.load('/fast/model_step71500_loss0.135.pt', map_location='cpu')
    config.coupled.model.load_state_dict(check['model_state_dict'],strict=True)
    for param in config.coupled.model.parameters():
        param.requires_grad = False
    """
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra = []
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra,is_output_only=True)

    #timesteps = [6,24]
    timesteps = [6,24,72]
    timesteps = [24, 48]
    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(30e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z=[0,6,12,18]
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=1024, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=10, lat_compress=False, timesteps=[24], train_timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=False, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(4,3,6)))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 5
    w.preserved_conf.save_every = 250
    w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.diff_loss = 5.
    w.preserved_conf.batch_size = 1
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 40_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1500
    w.preserved_conf.lr_sched.lr = 0.15e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'singing.fast': 4, 'stinson': 4},port=29500, start_method="spawn")
def Feb18_shallowpony():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)

    #timesteps = [6,24]
    timesteps = [6,24,72]
    timesteps = [24,72]
    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates
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


#@launch(nodes={'miramar.fast': 6, 'bimini.fast': 6, 'barceloneta':6},start_method='spawn')
#@launch(nodes={'singing.fast': 4, 'stinson': 4},port=29500, start_method="spawn")
#@launch(nodes={'stinson': 4},port=29500, start_method="spawn")
@launch(nodes={'miramar.fast': 6, 'bimini.fast': 6, 'barceloneta':6},start_method='spawn')
def Feb16_widepony_reprise():
    config.gpus = '0-5' #0,1,2,3,5
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)

    #timesteps = [6,24]
    timesteps = [6,24,72]
    timesteps = [24,72]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=928, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=10, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(4,3,6)))
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
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 40_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1500
    w.preserved_conf.lr_sched.lr = 0.15e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'bimini.fast': 6, 'miramar.fast': 6, 'barceloneta': 6},port=29500, start_method="spawn")
def Dec28_neoquadripede_neoft():
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1989, 6, 21), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    S0 = 12
    S0 = 6
    HH = 48
    HH = 24
    HH = 48
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[24,HH], max_ram_manual=int(6e9),
                                           worker_complain = False,
                                           only_at_z=[0,12],
                                           requested_dates = tdates,
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[24,HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=2, dec_swin_depth=2, proc_swin_depth=8, lat_compress=False, timesteps=[6,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = -1
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.save_every = 5
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.l2_loss = False
    w.preserved_conf.initial_gradscale = 65536.0 * 4
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    #w.preserved_conf.use_shampoo = True
    w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 20_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.0666e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'miramar.fast': 6, 'bimini.fast': 6, 'barceloneta':6},start_method='spawn')
def Feb8_shortking_replay1():
    config.gpus = '0-5' #0,1,2,3,5
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1990, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='fakeera5-28',extra_sfc_vars=extra, is_output_only=True)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)

    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z=[0,6,12,18]
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=1024, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=16, lat_compress=False, timesteps=[6], train_timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64, neorad=True))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.initial_gradscale = 65536.0 * 4
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 15_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1500
    w.preserved_conf.lr_sched.lr = 0.05e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


# 'singing.fast': 4, 
# this is also schopenhowitzer
@launch(nodes={'singing.fast': 4, 'stinson': 4},port=29500, start_method="spawn")
def Feb13_neolegeh24():
    config.gpus = '0-5'
    #config.nope = 1
    config.prefix = "_adapt"
    #config.name+= "-l1"
    #config.resume = "_"+config.name.replace("_","-")+"_"
    #config.resume = "run_Dec26-Luigi-l2_20231226-151561"

    SS = 6
    SS = 48
    SS = 24
    #SS = 72

    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25', input_levels=levels_gfs, levels=levels_medium)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1, mesh2],
                                           outputs = [mesh3],
                                           timesteps=[0,SS], max_ram_manual=int(4e9),
                                           requested_dates = get_dates((D(2021, 3, 20),D(2022, 12, 28))),
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
                train_timesteps=[0,SS],
                output_deltas = True,
                adapter_use_input_bias=True,
                use_matepoint = False,
                processor_dt = -1,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True),
                )
    
    adatper = model_unet.ForecastStepAdapterConv(model_conf)
    #from evals.package_neo import get_shorthegel
    #forecaster = get_shorthegel()
    from evals.package_neo import get_neoquadripede
    forecaster = get_neoquadripede()
    forecaster.config.use_matepoint = False
    model = model_unet.ForecastStepAdapterCombo(adatper,forecaster)
    # print number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1e6:0.1f}M")

    config.batch_size = 1
    config.use_neoloader = True
    config.ignore_train_safegaurd = True
    config.l2_loss = False
    config.validate_every = -1
    config.reset_optimizer = True
    config.log_every = 20
    config.save_every = 100
    config.initial_gradscale = 65536.0 * 2
    config.val_date_range =  [D(2022, 8, 1), D(2023, 1, 7)]
    config.lr_sched.cosine_period = 5_000
    config.lr_sched.cosine_bottom = 1e-7
    config.lr_sched.warmup_end_step = 300
    config.lr_sched.div_factor = 2
    config.lr_sched.lr = 0.8e-3
    config.optim = 'shampoo'
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.rerun_sample = 1
    config.yolo = False

    w = WeatherTrainer(data,conf=config,model=model)
    w.run()


@launch(nodes={'halfmoon':4},start_method='spawn')
def Jan31_legeh_nodeltas():
    config.gpus = '0-5'
    #config.nope = 1
    config.prefix = "_adapt"
    #config.name+= "-l1"
    config.resume = "_"+config.name.replace("_","-")+"_"
    #config.resume = "run_Dec26-Luigi-l2_20231226-151561"

    SS = 6
    SS = 24

    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25', input_levels=levels_gfs, levels=levels_medium)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1, mesh2],
                                           outputs = [mesh3],
                                           timesteps=[0,SS],
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
                timesteps=[0,SS], 
                output_deltas = False,
                adapter_use_input_bias=True,
                use_matepoint = False,
                processor_dt = -1,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True),
                )
    
    adatper = model_unet.ForecastStepAdapterConv(model_conf)
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
    config.validate_every = -1
    config.reset_optimizer = True
    config.log_every = 20
    config.save_every = 1000
    config.val_date_range =  [D(2022, 8, 1), D(2023, 1, 7)]
    config.lr_sched.cosine_period = 2_000
    config.lr_sched.cosine_bottom = 1e-7
    config.lr_sched.warmup_end_step = 250
    config.lr_sched.div_factor = 2
    config.lr_sched.lr = 1e-3 * 0.666
    config.optim = 'shampoo'
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.rerun_sample = 1
    config.yolo = False

    w = WeatherTrainer(data,conf=config,model=model)
    w.run()


@launch(nodes={'singing.fast': 4, 'stinson': 4},port=29500, start_method="spawn")
def Feb1_legeh_911_reprise():
    config.gpus = '0-5'
    #config.nope = 1
    config.prefix = "_adapt"
    #config.name+= "-l1"
    config.resume = "_"+config.name.replace("_","-")+"_"
    #config.resume = "run_Dec26-Luigi-l2_20231226-151561"

    SS = 6
    SS = 24

    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25', input_levels=levels_gfs, levels=levels_medium)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1, mesh2],
                                           outputs = [mesh3],
                                           timesteps=[0,SS],
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
                timesteps=[0,SS], 
                output_deltas = True,
                adapter_use_input_bias=True,
                use_matepoint = False,
                processor_dt = -1,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True),
                )
    
    adatper = model_unet.ForecastStepAdapterConv(model_conf)
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
    config.validate_every = -1
    config.reset_optimizer = True
    config.log_every = 20
    config.save_every = 100
    config.initial_gradscale = 65536.0 * 8
    config.val_date_range =  [D(2022, 8, 1), D(2023, 1, 7)]
    config.lr_sched.cosine_period = 5_000
    config.lr_sched.cosine_bottom = 1e-7
    config.lr_sched.warmup_end_step = 250
    config.lr_sched.div_factor = 2
    config.lr_sched.lr = 0.4e-3
    config.optim = 'shampoo'
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.rerun_sample = 1
    config.yolo = False

    w = WeatherTrainer(data,conf=config,model=model)
    w.run()


#@launch(nodes={'martins': 1},ddp=False, start_method="spawn")
#@launch(nodes={'singing.fast': 4, 'stinson': 4},port=29500, start_method="spawn")
@launch(nodes={'miramar.fast': 6, 'bimini.fast': 6, 'barceloneta':6},start_method='spawn')
def Jan29_shortking_reprise():
    config.gpus = '0-5' #0,1,2,3,5
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)

    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=1024, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=16, lat_compress=False, timesteps=[6], train_timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64, neorad=True))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 25_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    w.preserved_conf.lr_sched.lr = 0.25e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'halfmoon':4},start_method='spawn')
def Jan22_neohegel():
    config.gpus = '0-3'
    #config.nope = 1
    config.prefix = "_adapt"
    #config.name+= "-l1"
    #config.resume = "_"+config.name.replace("_","-")+"_"
    #config.resume = "run_Dec26-Luigi-l2_20231226-151561"

    SS = 6
    SS = 24

    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25', input_levels=levels_gfs, levels=levels_medium)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1, mesh2],
                                           outputs = [mesh3],
                                           timesteps=[0,SS],
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
                timesteps=[0,SS], 
                output_deltas = True,
                adapter_use_input_bias=True,
                use_matepoint = False,
                processor_dt = -1,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True),
                )
    
    adatper = model_unet.ForecastStepAdapterConv(model_conf)
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
    config.validate_every = -1
    #config.reset_optimizer = True
    config.log_every = 20
    config.save_every = 100
    config.val_date_range =  [D(2022, 8, 1), D(2023, 1, 7)]
    config.lr_sched.cosine_period = 12_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.div_factor = 2
    config.lr_sched.lr = 5e-4
    config.optim = 'shampoo'
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.rerun_sample = 1
    config.yolo = False

    w = WeatherTrainer(data,conf=config,model=model)
    w.run()


@launch(nodes={'halfmoon': 1},port=29500, start_method="spawn")
def Jan16_dbg():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    timesteps = [12,24,72]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[12,24], train_timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=64, neorad=True))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.use_shampoo = True
    #w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 31_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    w.preserved_conf.lr_sched.lr = 0.4e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'bimini.fast': 6, 'miramar.fast': 6, 'barceloneta': 6},port=29500, start_method="spawn")
def Jan16_handfist_neo2reprise():
    config.gpus = '0-5' #0,1,2,3,5
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    timesteps = [12,24,48]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=832, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[12,24], train_timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=64, neorad=True))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 30_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    w.preserved_conf.lr_sched.lr = 0.32e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


#@launch(nodes={'singing.fast': 4, 'stinson': 4},port=29500, start_method="spawn")
#@launch(nodes={'barceloneta': 6},port=29500, start_method="spawn")
@launch(nodes={'singing.fast': 4, 'stinson': 4},port=29500, start_method="spawn")
def Jan16_neocasio():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    timesteps = [1, 3, 6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps, max_ram_manual=int(5e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=True, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64, neorad=True))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.use_shampoo = True
    #w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 33_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000
    w.preserved_conf.lr_sched.lr = 0.33e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'bimini.fast': 6, 'barceloneta': 6},port=29500, start_method="spawn")
def Jan10_rolex():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    timesteps = [1, 3, 6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps, max_ram_manual=int(7e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24))
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,4,4), hidden_dim=384, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=4, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=True, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.use_shampoo = True
    #w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 20_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 500
    w.preserved_conf.lr_sched.lr = 0.33e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'miramar': 6},port=29500, start_method="spawn")
def Jan10_Enttauschung():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    timesteps = [12,24,72]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=832, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[12,24], train_timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=64))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 100
    w.preserved_conf.use_shampoo = True
    #w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 11_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 500
    w.preserved_conf.lr_sched.lr = 0.5e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'miramar': 6},port=29500, start_method="spawn")
def Jan9_reektest():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
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
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = 300_000_000
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 30_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.3e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()



@launch(nodes={'singing': 6},port=29500, start_method="spawn")
def Jan9_singingtest():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
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
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = 300_000_000
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 30_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.3e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()



@launch(nodes={'miramar.fast': 5, 'singing': 5},port=29500, start_method="spawn")
def Dec29_singlemenace_reprise():
    config.gpus = '0-5' #0,1,2,3,5
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
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
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1# 300_000_000
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 28_500
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.25e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()



@launch(ddp=1,nodes={'stinson.fast': 4, 'halfmoon':4},start_method='spawn')
#@launch(ddp=0,start_method='spawn')
def Dec30_L2BiasWaluigi():
    #config.gpus = '1,2,3,5'
    config.gpus = '0-3'
    #config.nope = 1
    config.prefix = "_adapt"
    #config.name += "-yolked3"
    #config.resume = "_"+config.name.replace("_","-")+"_"
    #config.resume = 'run_Dec27-Waluigi_20231227-212234'
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
                adapter_use_input_bias=True,
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
    config.l2_loss = True
    config.validate_every = 20
    config.log_every = 50
    config.save_every = 500
    config.val_date_range =  [D(2022, 8, 1), D(2023, 1, 7)]
    config.lr_sched.cosine_period = 20_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.div_factor = 2
    config.lr_sched.lr = 4.5e-4
    config.optim = 'shampoo' if am_i_torchrun() else 'adam' 
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.rerun_sample = 1
    config.yolo = False
    config.skip_audit = True

    w = WeatherTrainer(data,conf=config,model=model)
    w.run()


@launch(nodes={'miramar': 5},port=29500, start_method="spawn")
def Dec29_singlemenace_miramar():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
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
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = 300_000_000
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 30_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.3e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()



@launch(nodes={'singing': 5},port=29500, start_method="spawn")
def Dec29_singlemenace():
    config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
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
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = 300_000_000
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 30_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.3e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'bimini.fast': 6, 'barceloneta': 6},port=29500, start_method="spawn")
def Dec28_neoquadripede_beta72():
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1997, 6, 21), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    S0 = 12
    S0 = 6
    HH = 48
    HH = 24
    HH = 48
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[HH], max_ram_manual=int(4e9),
                                           worker_complain = False,
                                           only_at_z=[0,12],
                                           requested_dates = tdates,
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=2, dec_swin_depth=2, proc_swin_depth=8, lat_compress=False, timesteps=[6,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = -1
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.save_every = 5
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.l2_loss = True
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    #w.preserved_conf.use_shampoo = True
    w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 20_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.05e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'bimini.fast': 6, 'barceloneta': 6},port=29500, start_method="spawn")
def Dec28_neoquadripede():
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    S0 = 12
    S0 = 6
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[S0, 24], max_ram_manual=int(4.5e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[S0, 24], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=2, dec_swin_depth=2, proc_swin_depth=8, lat_compress=False, timesteps=[6,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 300_000_000
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.save_every = 5
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    #w.preserved_conf.use_shampoo = True
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.33e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()


@launch(nodes={'barceloneta': 1},port=29500, start_method="spawn")
def Dec28_devdec():
    config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[12, 24], max_ram_manual=int(6e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(1979, 1, 23),D(2017, 12, 28))),
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=128, enc_swin_depth=2, dec_swin_depth=2, proc_swin_depth=4, lat_compress=False, timesteps=[12, 24], dims_per_head=32, processor_dt=12, decoder_reinput_initial=True))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 1_000_000
    #w.preserved_conf.save_every = 5
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 31_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.33e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 1, 23),D(2017, 12,28)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 28)))
    w.run()


@launch(nodes={'singing':6},port=29500, start_method="spawn")
def Dec22_neoenc_ft():
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
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
    w.preserved_conf.use_shampoo = True
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
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1997, 1, 23),D(2017, 12,28)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 28)))
    w.run()


#@launch(nodes={'singing': 6},port=29500, start_method="spawn")
#@launch(nodes={'bimini.fast': 6, 'singing': 6},port=29500, start_method="spawn")
@launch(nodes={'bimini': 6},port=29500, start_method="spawn")
def Dec25_titanic():
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[24], max_ram_manual=int(6e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(1979, 1, 23),D(2017, 12, 28))),
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[24], dims_per_head=32, processor_dt=12))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 1_000_000
    #w.preserved_conf.save_every = 5
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 31_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.33e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 1, 23),D(2017, 12,28)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 28)))
    w.run()


@launch(ddp=1,nodes={'miramar':6})
def Dec19_ohp_l1hres_era5():
    #config.gpus = '1'
    config.nope = 1
    config.prefix = "_adapt"
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,levels=levels_tiny,source='hres')
    #dsc1 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5')
    dsc2 = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0,source='era5', levels=levels_tiny)
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
    w.preserved_conf.job = "adapter"
    w.run()


@launch(nodes={'barceloneta': 6},port=29500, start_method="spawn")
def Dec19_tiny():
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0, levels=levels_tiny)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), window_size=(2,6,12), checkpoint_every=1, hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32, surface_sandwich_bro=True, use_matepoint=False))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32, use_matepoint=False))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 250_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.25e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


@launch(nodes={'stinson.fast': 4, 'halfmoon': 4},port=29500, start_method="spawn")
def Dec18_singledec_hr():
    config.gpus = '0-3'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[24], max_ram_manual=int(6e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(1979, 1, 23),D(2017, 12, 28))),
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[24], dims_per_head=32, processor_dt=12))
    model.output_deltas = False
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 350_000
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



#@launch(nodes={'bimini.fast': 6, 'singing': 6},port=29500, start_method="spawn")
@launch(nodes={'barceloneta': 5},port=29500, start_method="spawn")
def Dec15_tardis_72hr_thaw():
    config.gpus = '0-4'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,72], max_ram_manual=int(6e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(1992, 1, 23),D(2017, 12, 28))),
                                           only_at_z=[0,6,12,18]
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,72], dims_per_head=32, only_at_z=[0,6,12,18]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 10
    w.preserved_conf.initial_gradscale = 65536.0 * 2
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    w.preserved_conf.l2_loss = True
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 20_000
    w.preserved_conf.lr_sched.lr = 0.125e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1992, 1, 23),D(2017, 12,28)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 28)))
    w.run()


#@launch(nodes={'martins': 1},port=29500, start_method="spawn")
@launch(nodes={'bimini.fast': 6, 'singing': 6},port=29500, start_method="spawn")
def Dec15_tardis_72hr():
    config.gpus = '0'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,72], max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(1992, 1, 23),D(2017, 12, 28))),
                                           only_at_z=[0,6,12,18]
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,72], dims_per_head=32, only_at_z=[0,6,12,18]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.initial_gradscale = 65536.0 * 4
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    w.preserved_conf.l2_loss = False
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 175_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 15_000
    w.preserved_conf.lr_sched.lr = 0.25e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1992, 1, 23),D(2017, 12,28)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 28)))
    w.run()

@launch(nodes={'singing': 6},port=29500, start_method="spawn")
def Dec19_tiny_ft():
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    dsc = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, levels=levels_tiny, source='era5-13')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[dsc], outputs=[dsc],
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(2000, 1, 23),D(2017, 12,30))),
                                           only_at_z=[0,6,12,18]
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(2,8,8), window_size=(2,6,12), checkpoint_every=1, hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32, surface_sandwich_bro=True, use_matepoint=False))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32, use_matepoint=False))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.initial_gradscale = 65536.0 * 4
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.reset_optimizer = True
    w.preserved_conf.l2_loss = True
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 50_000
    w.preserved_conf.lr_sched.warmup_end_step = 2_000
    w.preserved_conf.lr_sched.lr = 0.02e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(2000, 1, 23),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


@launch(nodes={'bimini.fast': 6, 'singing': 6},port=29500, start_method="spawn")
def Dec16_smolmenace_ft():
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0, levels=levels_ecm2)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(2000, 1, 23),D(2017, 12,30))),
                                           only_at_z=[0,6,12,18]
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.initial_gradscale = 65536.0 * 8
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.l2_loss = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 250_000 // 10
    w.preserved_conf.lr_sched.warmup_end_step = 10_000 // 10
    w.preserved_conf.lr_sched.lr = 0.025e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(2000, 1, 23),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


@launch(nodes={'bimini.fast': 6, 'singing': 6},port=29500, start_method="spawn")
def Dec16_smolmenace_reprise():
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0, levels=levels_ecm2)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 350_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.15e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()



#@launch(nodes={'miramar.fast': 5, 'barceloneta': 5},port=29500, start_method='spawn')
@launch(nodes={'bimini.fast': 6, 'singing': 6},port=29500, start_method="spawn")
def Dec16_smolmenace():
    config.gpus = '0-5'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    #config.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0, levels=levels_ecm2)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 350_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.3e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


#@launch(nodes={'stinson.fast': 4, 'halfmoon': 4},port=29500, start_method='spawn')
@launch(nodes={'miramar.fast': 6, 'barceloneta': 6},port=29500, start_method='spawn')
def Dec18_midmenace():
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0, levels=levels_ecm2)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[6,24], max_ram_manual=int(9e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=1024, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 450_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.175e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()




#@launch(nodes={'stinson': 1},port=29500, start_method='spawn')
@launch(nodes={'stinson.fast': 4, 'halfmoon': 4},port=29500, start_method='spawn')
def Dec14_ens():
    config.gpus = '0-3'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    #config.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0, levels=levels_ecm2)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 250_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.3e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


@launch(nodes={'bimini.fast': 6, 'singing': 6},port=29500)
def Dec4_tardis_neol2ft_reprise():
    args.gpus = '0-5'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(2000, 1, 23),D(2017, 12,30))),
                                           only_at_z=[0,6,12,18]
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32, only_at_z=[0,6,12,18]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args=args,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.initial_gradscale = 65536.0 * 4
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    w.preserved_conf.l2_loss = True
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 250_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 20_000
    w.preserved_conf.lr_sched.lr = 0.025e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(2000, 1, 23),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


@launch(nodes={'bimini.fast': 6, 'singing': 6},port=29500)
def Dec4_tardis_slowl2ft():
    args.gpus = '0-5'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(2000, 1, 23),D(2017, 12,30))),
                                           only_at_z=[0,6,12,18]
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32, only_at_z=[0,6,12,18]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args=args,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.initial_gradscale = 65536.0 * 4
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    w.preserved_conf.l2_loss = True
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 250_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 15_000
    w.preserved_conf.lr_sched.lr = 0.0125e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(2000, 1, 23),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()

#@launch(nodes={'stinson.fast': 4, 'halfmoon': 4},port=29500)
@launch(nodes={'miramar.fast': 6, 'barceloneta': 6},port=29500)
def Dec12_quadripede():
    args.gpus = '0-5'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[6,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=2, dec_swin_depth=2, proc_swin_depth=8, lat_compress=False, timesteps=[6,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args=args,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.25e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()



@launch(nodes={'miramar.fast': 6, 'barceloneta': 6},port=29500)
def Dec4_tardis_l2ft():
    args.gpus = '0-5'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(2000, 1, 23),D(2017, 12,30))),
                                           only_at_z=[0,12]
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32, only_at_z=[0,12]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args=args,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    w.preserved_conf.l2_loss = True
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
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


@launch(nodes={'stinson.fast': 4, 'halfmoon': 4},port=29500)
def Dec11_ihmlawtd():
    args.gpus = '0-3'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=1024, enc_swin_depth=2, dec_swin_depth=2, proc_swin_depth=12, lat_compress=not False, timesteps=[12,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args=args,model=model,data=data)
    w.preserved_conf.validate_every = 200
    #w.preserved_conf.validate_every = 10
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.25e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()

@launch(nodes={'stinson.fast': 4, 'halfmoon': 4},port=29500)
def Dec4_tardis_reprise():
    args.gpus = '0-3'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args=args,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 250_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.25e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()

@launch(nodes={'bimini.fast': 6, 'singing': 6},port=29500)
def Dec12_doubledouble_protein():
    args.gpus = '0-5'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args=args,model=model,data=data)
    w.preserved_conf.validate_every = 300
    w.preserved_conf.save_every = 300
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 350_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.4e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


@launch(nodes={'miramar.fast': 6, 'barceloneta': 6},port=29500)
def Dec12_doubledouble_animal():
    args.gpus = '0-5'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args=args,model=model,data=data)
    w.preserved_conf.validate_every = 300
    w.preserved_conf.save_every = 300
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = True
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 350_000
    w.preserved_conf.lr_sched.cosine_bottom = 1e-7
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.3e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()



@launch(nodes={'stinson.fast': 4, 'halfmoon': 4},port=29500)
def Dec4_tardis():
    args.gpus = '0-3'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,model,data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 350_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.3e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


#'singing.fast': 6, 'miramar.fast': 6, 'bimini.fast': 6, 
@launch(nodes={'singing.fast': 6, 'miramar.fast': 6, 'bimini.fast': 6, 'barceloneta': 6},port=29500)
def Dec4_neoreal_neoreprise():
    args.gpus = '0-5'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[24], max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(2000, 1, 23),D(2017, 12,30))),
                                           only_at_z=[0,12]
                                           ))
    print("uhhh data.config is", data.config)
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=28, lat_compress=False, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args=args,model=model,data=data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 350_000
    w.preserved_conf.lr_sched.warmup_end_step = 20_000
    w.preserved_conf.lr_sched.lr = 0.0666e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(2000, 1, 23),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()



@launch(nodes={'singing.fast': 6, 'miramar.fast': 6, 'barceloneta': 6},port=29500)
def Dec4_neoreal_reprise():
    args.gpus = '0-5'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[24], max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=28, lat_compress=False, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,model,data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 350_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.2e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()



@launch(nodes={'singing.fast': 6, 'miramar.fast': 6, 'barceloneta': 6},port=29500)
def Dec4_neoreal():
    args.gpus = '0-5'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[24], max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=28, lat_compress=False, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,model,data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = False
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.lr_sched.warmup_end_step = 15_000
    w.preserved_conf.lr_sched.lr = 0.2e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


# 'stinson.fast': 4, 'singing.fast': 4, 
@launch(nodes={'stinson.fast': 4, 'singing.fast': 4, 'halfmoon': 4},port=29500)
def Dec3_dove():
    args.gpus = '0-3'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=28, lat_compress=True, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,model,data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 400_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.25e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()

@launch(nodes={'stinson.fast': 4, 'singing.fast': 4, 'halfmoon': 4},port=29500)
def Dec3_dove_ft():
    args.gpus = '0-3'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=28, lat_compress=True, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,model,data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.reset_optimizer = True
    w.preserved_conf.use_shampoo = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.2e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()



@launch(nodes={'miramar.fast': 6, 'barceloneta': 6},port=29500)
def Dec3_pantene_ft():
    args.gpus = '0-5'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1997, 6, 21),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=24, lat_compress=False, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,model,data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.125e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1997, 6, 21),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


@launch(nodes={'miramar.fast': 6, 'barceloneta': 6},port=29500)
def Dec3_pantene():
    args.gpus = '0-5'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=24, lat_compress=False, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,model,data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 400_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.25e-3 * 0.666
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1979, 3, 28),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


@launch(nodes={'miramar': 6},port=29500)
def Dec1_shampoo():
    args.gpus = '0-5'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1971, 1, 1),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=24, lat_compress=False, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,model,data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    w.preserved_conf.use_shampoo = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 500_000
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    w.preserved_conf.lr_sched.lr = 0.25e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


@launch(nodes={'halfmoon.fast': 4, 'stinson.fast': 4, 'barceloneta':6},port=29500)
def Nov24_byrne():
    args.gpus = '0-5'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(ForecastStepConfig(mesh, patch_size=(4,8,8), hidden_dim=1024, proc_swin_depth=16, lat_compress=True, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.lr_sched.warmup_end_step = 5_000
    w.preserved_conf.lr_sched.lr = 0.1e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1997, 6, 21),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


@launch(nodes={'halfmoon.fast': 4, 'stinson.fast': 4, 'barceloneta':6},port=29500)
def Nov18_peters():
    args.gpus = '0-5'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=36, lat_compress=True)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 12
    w.preserved_conf.log_every = 50
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 600_000
    w.preserved_conf.lr_sched.warmup_end_step = 5_000
    w.preserved_conf.lr_sched.lr = 0.05e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = -711994
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1997, 6, 21),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()

"""
@launch(nodes={'halfmoon.fast': 4, 'stinson.fast': 4, 'barceloneta':6},port=29500)
def Nov18_peters():
    args.gpus = '0-5'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    #args.nope = True
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=36, lat_compress=True)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 12
    w.preserved_conf.log_every = 50
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 1_200_000
    w.preserved_conf.lr_sched.warmup_end_step = 5_000
    w.preserved_conf.lr_sched.lr = 0.05e-3
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = get_dates((D(1997, 6, 21),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()
    """


@launch(nodes={'singing':3},port=29500)
def Nov16_latcomp():
    args.gpus = '0,3,5'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, lat_compress=True)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = [12]
    w.preserved_conf.DH = 24
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.lr_sched.lr = 0.05e-3  
    w.run()

@launch(nodes={'singing':3},port=29501,kill_nvidia=False)
def Nov16_nolatcomp():
    args.gpus = '1,2,4'
    args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, lat_compress=False)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = [12]
    w.preserved_conf.DH = 24
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.lr_sched.lr = 0.05e-3  
    w.run()

@launch(nodes={'rockaway':3},port=29500)
def Nov17_lite2():
    #args.gpus = '0-5'
    args.gpus = '0,2,3'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, lat_compress=True)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = [12]
    w.preserved_conf.DH = 24
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.run()



@launch(nodes={'singing':6},port=29500)
def Nov17_lite():
    args.gpus = '0-5'
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, lat_compress=True)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = [12]
    w.preserved_conf.DH = 24
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.run()


@launch(ddp=False,nodes={'martins':1})
def Nov21_compound():
    args.gpus = '0'
    args.nope = True
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    conf = ForecastStepConfig(mesh,patch_size=(4,8,8),hidden_dim=768,depth=24,lat_compress=True,
                              timesteps=[12,24])
    model = ForecastStepSwin3D(conf)
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 20
    #w.preserved_conf.only_at_z = [12]
    w.preserved_conf.dates = get_dates((D(2016, 1, 1),D(2016, 1,7)))
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.lr_sched.lr = 0.1e-3  
    w.run()

@launch(nodes={'singing':6},port=29500,kill_nvidia=True)
def Nov21_12and24():
    args.gpus = '0-5'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"

    timesteps = [12,24]
    dsc = NeoDatasetConfig(WEATHERBENCH=1)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=timesteps,
                                           requested_dates = get_dates((D(1997, 1, 1),D(2017, 12,1))),
                                           use_mmap = 1,
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh,patch_size=(4,8,8),hidden_dim=768,lat_compress=True,
                            timesteps=timesteps,processor_dt=12))
    w = WeatherTrainer(args,model,data)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.log_every = 20
    w.preserved_conf.save_every = 85
    w.preserved_conf.only_at_z = None
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.lr_sched.warmup_end_step = 5_000
    w.preserved_conf.lr_sched.lr = 2*5e-6
    w.preserved_conf.lr_sched.step_offset = -int(3.273e5)
    w.run()


@launch(nodes={'singing':3},port=29501)
def Nov21_just24():
    args.gpus = '0-2'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    conf = ForecastStepConfig(mesh,patch_size=(4,8,8),hidden_dim=768,lat_compress=True,
                            timesteps=[24],processor_dt=12)
    model = ForecastStepSwin3D(conf)
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.validate_every = 200
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 20
    #w.preserved_conf.only_at_z = [12]
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.lr_sched.lr = 0.1e-3  
    w.run()

@launch(ddp=False,nodes={'martins':1})
def Nov23_dataload():
    args.gpus = '0'
    args.nope = True
    timesteps = [24]
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    data = NeoWeatherDataset(NeoDataConfig(mesh=mesh,timesteps=timesteps,CLOUD=False,
                                           requested_dates= get_dates((D(2016, 1, 1),D(2016, 1,3)))
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(mesh,patch_size=(4,8,8),hidden_dim=768,lat_compress=True,
                            timesteps=timesteps,processor_dt=12))
    w = WeatherTrainer(args,mesh,model,data)
    w.preserved_conf.validate_every = 1
    w.preserved_conf.log_every = 2
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.lr_sched.lr = 0.05e-3  
    w.run()

@launch(ddp=False,nodes={'martins':1})
def Nov23_random():
    args.gpus = '0'
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    data = NeoWeatherDataset(NeoDataConfig(mesh=mesh,timesteps=[6,12,18,24]))
    model = ForecastStepSwin3D(ForecastStepConfig(mesh,timesteps=[6,12,18,24],return_random=True))
    w = WeatherTrainer(args,mesh,model,data)
    w.conf.log_every = 1
    w.conf.validate_every = 1e9
    w.run()

if __name__ == '__main__':
    run(locals().values())
