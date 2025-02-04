from launch import *

if am_i_torchrun():
    from train import *

from train import * # <--- comment this back in if not using DDP
from torch.utils import checkpoint as torch_checkpoint



@launch(nodes={'muir': 6} ,port=29504, start_method="spawn",kill_nvidia=False, zulip=True, ping='@**John Dean**',validate=False)
def Oct11_chief():
    #config.nope = True
    #config.gpus='1,2,3,4,5,6'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '142_lsp', '143_cp', '168_2d', '201_mx2t', '202_mn2t']
    extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)

    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=[0], 
                                           requested_dates = tdates,
                                           ))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            Transformer=SlideLayer3D,
            Encoder=ForecastStepUEncoder,
            Decoder=ForecastStepUDecoder,
            ucodec_config = UCodecConfig(
                conv_sz = [(2,4,4), (2,2,2)],
                conv_dim = [200, 1120],
                tr_headdim = [40,40],
                tr_win = (5,3,3),
                tr_depth = [2, 4],
                blend_conv = True,
            ),
            patch_size=(4,8,8),
            hidden_dim=1120,
            proc_swin_depth=10,
            dims_per_head=40,
            processor_dt=6,
            sincos_latlon = True,
            neorad=True,
            window_size=(5,5,7)))
    params = sum(p.numel() for p in model.encoder.parameters())
    print(f"ENCODER PARAMS: {params/1e6:.2f}M")

    params = sum(p.numel() for p in model.decoders['0'].parameters())
    print(f"DECODER PARAMS: {params/1e6:.2f}M")

    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 250
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.restart_warmup_end_step = 1000
    config.lr_sched.lr = 0.3e-3 * 0.75
    config.lr_sched.schedule_dts = True
    config.save_imgs_every = 25
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'barceloneta': 6} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Sep22_ucodec_blend_nopoles():

    #config.nope = True
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, bro_zero_me_out=3, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    #timesteps = [0,6,12,18,24]
    timesteps = [6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           random_timestep_subset = 2
                                           ))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            #checkpointfn = checkpoint.checkpoint,
            output_deltas=False,
            Transformer=SlideLayer3D,
            Encoder=ForecastStepUEncoder,
            Decoder=ForecastStepUDecoder,
            ucodec_config = UCodecConfig(
                conv_sz = [(2,4,4), (2,2,2)],
                conv_dim = [192, 896],
                tr_headdim = [32, 32],
                tr_win = (7,5,5),
                tr_depth = [2, 4],
                blend_conv = True,
            ),
            patch_size=(4,8,8),
            hidden_dim=896,
            #hidden_dim=1024,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=6,
            sincos_latlon = True,
            neorad=True,
            window_size=(7,9,9)))
    params = sum(p.numel() for p in model.encoder.parameters())
    print(f"ENCODER PARAMS: {params/1e6:.2f}M")

    params = sum(p.numel() for p in model.decoders['0'].parameters())
    print(f"DECODER PARAMS: {params/1e6:.2f}M")


    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.schedule_dts = True
    config.use_neoloader = False
    config.save_imgs_every = 25
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(nodes={'barceloneta': 2},port=29501, start_method="spawn")#, ddp=0)
@launch(ddp=0,start_method='spawn')
def Sep30_doctorram():
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.disregard_buffer_checksum = False#True
    config.nope = False
    # tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 4, 30))])
    extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    # imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra_all, output_only_vars=extra_out_only, input_levels=levels_medium, levels=levels_joank)
    # omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra_all, output_only_vars=extra_out_only, input_levels=levels_medium, levels=levels_joank)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28', extra_sfc_vars=extra_all, bro_zero_me_out=len(extra_out_only), input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28', extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6, 12, 18, 24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(6e9),
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
            #matepoint_table='auto',
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            dims_per_head=32, 
            processor_dt=3, 
            output_deltas=False, 
            decoder_reinput_initial=False,
            neorad=True,
            checkpoint_convs = True, 
        window_size=(3,5,7)))

    config.HALF = True
    config.ignore_train_safegaurd = True
    config.validate_every = -1
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 1024
    #config.optim = 'adam'
    config.reset_optimizer = False
    #config.lr_sched = SimpleNamespace()
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
    config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'barceloneta': 6} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Sep30_chunkfix_ddp_matepoint_624():
    #torch.autograd.set_detect_anomaly(True)
    #config.nope = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, bro_zero_me_out=3, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            #checkpointfn = checkpoint.checkpoint,
            output_deltas=False,
            Transformer=SlideLayer3D,
            Encoder=ForecastStepUEncoder,
            Decoder=ForecastStepUDecoder,
            ucodec_config = UCodecConfig(
                conv_sz = [(2,4,4), (2,2,2)],
                conv_dim = [192, 896],
                tr_headdim = [32, 32],
                tr_checkpoint_chunks = [2,1],
                tr_win = (7,5,5),
                tr_depth = [2, 4],
            ),
            patch_size=(4,8,8),
            hidden_dim=896,
            #hidden_dim=1024,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=6,
            sincos_latlon = True,
            neorad=True,
            window_size=(7,9,9)))

    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.schedule_dts = False
    config.use_neoloader = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'miramar': 6} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
#@launch(nodes={'barceloneta': 6} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Sep30_ucodec_neoschedule():

    #config.nope = True
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates
                                           ))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            Encoder=ForecastStepUEncoder,
            Decoder=ForecastStepUDecoder,
            ucodec_config = UCodecConfig(
                conv_sz = [(2,4,4), (2,2,2)],
                conv_dim = [192, 896],
                tr_headdim = [32, 32],
                tr_win = (7,5,5),
                tr_depth = [2, 4],
                tr_checkpoint_chunks = [2,1],
            ),
            patch_size=(4,8,8),
            hidden_dim=896,
            #hidden_dim=1024,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=6,
            sincos_latlon = True,
            neorad=True,
            window_size=(7,9,9)))

    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.schedule_dts = True
    config.use_neoloader = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'miramar': 6} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Sep29_dualprocs():

    #config.nope = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, bro_zero_me_out=3, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    timesteps = [3,6,9,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            Encoder=ForecastStepUEncoder,
            Decoder=ForecastStepUDecoder,
            ucodec_config = UCodecConfig(
                conv_sz = [(2,4,4), (2,2,2)],
                conv_dim = [192, 896],
                tr_headdim = [32, 32],
                tr_win = (7,5,5),
                tr_depth = [2, 4],
            ),
            patch_size=(4,8,8),
            hidden_dim=896,
            #hidden_dim=1024,
            dims_per_head=32,
            proc_depths = [4,8],
            processor_dt=[1,6],
            sincos_latlon = True,
            neorad=True,
            window_size=(7,9,9)))


    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 0.3e-3
    config.use_neoloader = False
    config.save_imgs = True
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'bimini': 6} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Sep22_ucodec():

    #config.nope = True
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, bro_zero_me_out=3, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            Encoder=ForecastStepUEncoder,
            Decoder=ForecastStepUDecoder,
            ucodec_config = UCodecConfig(
                conv_sz = [(2,4,4), (2,2,2)],
                conv_dim = [192, 896],
                tr_headdim = [32, 32],
                tr_win = (7,5,5),
                tr_depth = [2, 4],
            ),
            patch_size=(4,8,8),
            hidden_dim=896,
            #hidden_dim=1024,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=6,
            sincos_latlon = True,
            neorad=True,
            window_size=(7,9,9)))
    params = sum(p.numel() for p in model.encoder.parameters())
    print(f"ENCODER PARAMS: {params/1e6:.2f}M")

    params = sum(p.numel() for p in model.decoders['0'].parameters())
    print(f"DECODER PARAMS: {params/1e6:.2f}M")


    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 0.3e-3
    config.use_neoloader = False
    config.save_imgs = True
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0,start_method='spawn')
@launch(nodes={'muir': 6} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=True)
def Sep22_hann():

    #config.nope = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    timesteps = [0,6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            Encoder=ForecastStepUEncoder,
            Decoder=ForecastStepUDecoder,
            ucodec_config = UCodecConfig(
                conv_sz = [(2,3,3), (1,3,3), (2,2,2)],
                conv_dim = [64, 256, 1152],
                tr_headdim = [16, 32, 32],
                tr_win = (3,3,3),
                tr_depth = [2, 2, 4],
            ),
            patch_size=(4,18,18),
            hidden_dim=1152,
            #hidden_dim=1024,
            enc_swin_depth=8,
            dec_swin_depth=8,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=6,
            neorad=True,
        window_size=(5,5,5)))
    params = sum(p.numel() for p in model.encoder.parameters())
    print(f"ENCODER PARAMS: {params/1e6:.2f}M")

    params = sum(p.numel() for p in model.decoders['0'].parameters())
    print(f"DECODER PARAMS: {params/1e6:.2f}M")

    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 0.3e-3
    config.use_neoloader = False
    config.profile = True
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0,start_method='spawn')
@launch(nodes={'barceloneta': 6} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=True)
def Sep21_hamming():

    #config.nope = True
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    timesteps = [0,6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            Encoder=ForecastStepNeoEncoder,
            Decoder=ForecastStepNeoDecoder,
            patch_size=(4,16,16),
            hidden_dim=1280,
            #hidden_dim=1024,
            enc_swin_depth=8,
            dec_swin_depth=8,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=6,
            neorad=True,
        window_size=(3,5,7)))
    params = sum(p.numel() for p in model.encoder.parameters())
    print(f"ENCODER PARAMS: {params/1e6:.2f}M")

    params = sum(p.numel() for p in model.decoders['0'].parameters())
    print(f"DECODER PARAMS: {params/1e6:.2f}M")

    config.ignore_train_safegaurd = True
    config.auto_loss_consts = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 0.3e-3
    config.use_neoloader = False
    config.profile = True
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(ddp=0,start_method='spawn')
def Jul29_bachelor():

    config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(8e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
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

    params = sum(p.numel() for p in model.encoder.parameters())
    print(f"ENCODER PARAMS: {params/1e6:.2f}M")

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
    config.use_neoloader = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

if __name__ == '__main__':
    run(locals().values())
