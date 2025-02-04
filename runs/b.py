from launch import *

if am_i_torchrun():
    from train import *

from train import * # <--- comment this back in if not using DDP
from torch.utils import checkpoint as torch_checkpoint

#@launch(nodes={'barceloneta': 6},port=29505, start_method="spawn", zulip=True, ping='@**Haoxing Du**',validate=False,kill_nvidia=True)
@launch(ddp=0,start_method='spawn')
def Sep24_YB_serp3rand144hr_rtft_dltest():
    #torch.autograd.set_detect_anomaly(True)
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
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
    
    timesteps = [0,3,6,9,24,48]
    data = WeatherDataset(DataConfig(inputs=[mesh1,mesh2], outputs=[omesh],
                                           timesteps=timesteps, 
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

#@launch(nodes={'muir': 4},port=29505)
@launch(ddp=0,start_method='spawn')
def Sep26_YBft_rand144hr_long_dltest():
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    import evals.package_neo as pn
    model = pn.get_yamahabachelor0()
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    timesteps = [0] + list(range(1,25,1)) + list(range(28,49,4)) + list(range(54,97,6)) + list(range(108,145,12))
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
                                           only_at_z = list(range(24)),
                                           requested_dates = tdates,
                                           random_timestep_subset = 6
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
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 10_000
    config.lr_sched.cosine_bottom = 5e-9
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    config.lr_sched.lr = 5e-5
    config.use_neoloader = False
    config.lr_sched.step_offset = 0
    config.auto_loss_consts = True
    config.print_ram_usage = True
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(nodes={'ip-172-31-3-232': 1} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
@launch(nodes={'muir': 6} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def iweep():

    config.nope = True
    #config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    load_locations = ['/jersey/','/fast/proc/']
    extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '142_lsp', '143_cp', '168_2d', '201_mx2t', '202_mn2t']
    extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(load_locations=load_locations,source='era5-28', extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(load_locations=load_locations,source='era5-28', extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    timesteps = [0,6] 
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates,
                                           random_timestep_subset = 2
                                           ))
    config.lr_sched.schedule_dts = True

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
                conv_dim = [256, 1536],
                tr_headdim = [32, 32],
                tr_win = (7,5,5),
                tr_depth = [2, 4],
                blend_conv = True,
            ),
            patch_size=(4,8,8),
            hidden_dim=1536,
            #hidden_dim=1024,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=6,
            sincos_latlon = True,
            neorad=True,
            window_size=(7,9,9)))

    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.HALF = True
    config.use_tf32 = True
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 0.3e-3
    config.save_imgs_every = 25
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


#@launch(nodes={'ip-172-31-3-232': 1} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
#@launch(nodes={'muir': 6} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
@launch(ddp=1,start_method='spawn')
def icry():

    config.nope = True
    #config.gpus = '2,3,4,5,6,7'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    #load_locations = ['/jersey/','/fast/proc/']
    load_locations = ['/fast/proc/', '/jersey/']
    extra_in_out = ['15_msnswrf', '45_tcc', '168_2d', '034_sstk', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(load_locations=load_locations,source='era5-28', extra_sfc_vars=extra_in_out, extra_sfc_pad=3, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(load_locations=load_locations,source='era5-28', extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    timesteps = [0,6] 
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates,
                                           random_timestep_subset = 2
                                           ))
    config.lr_sched.schedule_dts = True
    
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

    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.HALF = True
    config.use_tf32 = True
    config.optim = 'shampoo'
    config.optim = 'adam'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.restart_warmup_end_step = 500
    config.save_imgs_every = 25
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(nodes={'ip-172-31-3-232': 8} ,port=29504, start_method="spawn", zulip=True, ping='@**John Dean**',validate=False)
@launch(ddp=0,start_method='spawn')
def noneo():

    config.nope = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, bro_zero_me_out=3, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    #timesteps = [0,6,12,18,24]
    timesteps = [6]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
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


#@launch(nodes={'ip-172-31-3-232': 8} ,port=29504, start_method="spawn")#, zulip=True, ping='@**John Dean**',validate=False)
@launch(ddp=0,start_method='spawn')
def cloudsux():

    config.nope = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #extra = ['034_sstk', 'logtp', '15_msnswrf', '45_tcc']
    load_locations = ['/jersey/','/fast/proc/']
    imesh = meshes.LatLonGrid(load_locations=load_locations, source='era5-28',extra_sfc_vars=extra, bro_zero_me_out=3, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(load_locations=load_locations, source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)

    #timesteps = [0,6,12,18,24]
    timesteps = [0,6,12,18,24]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
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
    config.save_imgs_every = 25
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


if __name__ == '__main__':
    run(locals().values())
