import sys
sys.path.append('.')
from runs.launch import *

if am_i_torchrun():
    from train.trainer import *

from train.trainer import * # <--- comment this back in if not using DDP
from datasets import *

@launch(nodes={'muir': 6},port=29502, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')
#@launch(ddp=False)
def Jan15_warwick():
    #config.gpus = '2'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1971, 1, 1), D(2019, 12, 31)), (D(2021, 1, 1), D(2024, 1, 1))])
    extra_in = []#'45_tcc', '034_sstk', '168_2d', '136_tcw', '137_tcwv']
    extra_out = extra_in + ['45_tcc', '034_sstk', '168_2d', '136_tcw', '137_tcwv', '179_ttr', '142_lsp', '143_cp', 'logtp', '201_mx2t', '202_mn2t']
    imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_in, input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_out, input_levels=levels_medium, levels=levels_ecm1)
    data = WeatherDataset(DataConfig(inputs=[DailyAnalysisDataset(imesh)],
                                        outputs=[DailyAnalysisDataset(omesh)],
                                        requested_dates = tdates,
                                        only_at_z = [0],
                                        random_timestep_subset = 4,
                                        ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.decoder import SimpleConvDecoder
    conf = ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        #encdec_tr_depth=4,
        #oldpr=True,
        #window_size=(3,7,7),
        #patch_size=(2,8,8),
        simple_decoder_patch_size=(1,8,8),
        latent_size=768,
        #pr_dims = [48, 192, 768],
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
    )
    conf.slow_start = True

    model = ForecastModel(conf,
                          encoders=[SimpleConvEncoder(imesh,conf)],
                          decoders=[SimpleConvDecoder(omesh,conf)])
    print(model)

    print_total_params(model)

    #config.timesteps = timesteps
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 20_000
    config.latent_l2 = 1e-4
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.optim = 'shampoo'
    #config.shampoo.version = 'new'
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched.schedule_dts = True
    #config.lr_sched.steps_till_max_dt_max = 5_000 
    #config.lr_sched.num_random_subset_max = 5
    config.lr_sched.max_dt_min = 24*3
    config.lr_sched.max_dt_max = 24*21
    config.lr_sched.cosine_period = 10_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.cosine_bottom = 5e-8
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    config.save_imgs_every = 100
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


#@launch(nodes={'muir': 6},port=29502, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')
@launch(ddp=False)
def Jan11_fenrir():
    #config.gpus = '2'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    tdates = get_dates([(D(1971, 1, 1), D(2019, 12, 31)), (D(2021, 1, 1), D(2024, 1, 1))])
    extra_in = ['45_tcc', '034_sstk', '168_2d', '136_tcw', '137_tcwv']
    extra_out = extra_in + ['179_ttr', '142_lsp', '143_cp', 'logtp', '201_mx2t', '202_mn2t']
    #imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=[], input_levels=levels_medium)
    imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_in, input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_out, input_levels=levels_medium, levels=levels_smol)
    data = WeatherDataset(DataConfig(inputs=[DailyAnalysisDataset(imesh)],
                                        outputs=[DailyAnalysisDataset(omesh)],
                                        requested_dates = tdates,
                                        only_at_z = [0],
                                        random_timestep_subset = 4,
                                        ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.decoder import SimpleConvDecoder
    conf = ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        encdec_tr_depth=4,
        #oldpr=True,
        #window_size=(3,7,7),
        #patch_size=(2,8,8),
        simple_decoder_patch_size=(1,8,8),
        latent_size=768,
        #pr_dims = [48, 192, 768],
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
    )
    conf.slow_start = True

    model = ForecastModel(conf,
                          encoders=[SimpleConvEncoder(imesh,conf)],
                          decoders=[SimpleConvDecoder(omesh,conf)])
    print(model)

    print_total_params(model)

    #config.timesteps = timesteps
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 20_000
    config.latent_l2 = 1e-4
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.optim = 'shampoo'
    #config.shampoo.version = 'new'
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched.schedule_dts = True
    #config.lr_sched.steps_till_max_dt_max = 5_000 
    #config.lr_sched.num_random_subset_max = 5
    config.lr_sched.max_dt_min = 24*3
    config.lr_sched.max_dt_max = 24*21
    config.lr_sched.cosine_period = 15_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.cosine_bottom = 5e-8
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    config.save_imgs_every = 100
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'muir': 6},port=29502, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')
#@launch(ddp=False, kill_nvidia=False)
def Dec30_tyr():
    #config.gpus = '2'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(2008, 1, 1), D(2019, 12, 31)), (D(2021, 1, 1), D(2024, 7, 31))])
    extra_in = ['45_tcc', '034_sstk', '168_2d', '136_tcw', '137_tcwv']
    extra_out = extra_in + ['179_ttr', '142_lsp', '143_cp', 'logtp', '201_mx2t', '202_mn2t']
    #imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=[], input_levels=levels_medium)
    imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_in, input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_out, input_levels=levels_medium, levels=levels_smol)
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           requested_dates = tdates,
                                           only_at_z = [0],
                                           random_timestep_subset = 4,
                                           ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.decoder import SimpleConvDecoder
    conf = ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        encdec_tr_depth=4,
        #oldpr=True,
        #window_size=(3,7,7),
        #patch_size=(2,8,8),
        simple_decoder_patch_size=(1,8,8),
        latent_size=768,
        #pr_dims = [48, 192, 768],
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
    )
    conf.slow_start = True

    model = ForecastModel(imesh,conf,
                          encoders=[SimpleConvEncoder(imesh,conf)],
                          decoders=[SimpleConvDecoder(omesh,conf)])
    print(model)

    print_total_params(model)

    #config.timesteps = timesteps
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 20_000
    config.latent_l2 = 1e-4
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.optim = 'shampoo'
    #config.shampoo.version = 'new'
    #config.optim = 'adam'
    config.reset_optimizer = True #False
    config.lr_sched.schedule_dts = True
    #config.lr_sched.steps_till_max_dt_max = 5_000 
    #config.lr_sched.num_random_subset_max = 5
    config.lr_sched.max_dt_min = 24*3
    config.lr_sched.max_dt_max = 24*21
    config.lr_sched.cosine_period = 15_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.cosine_bottom = 5e-8
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    config.save_imgs_every = 100
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'stinson': 4},port=29502, start_method="spawn", kill_nvidia=False)#, zulip=True, ping='@**Anuj Shetty**')
#@launch(ddp=False, kill_nvidia=False)
def Dec13_testing():
    #config.gpus = '2'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(2008, 1, 1), D(2019, 12, 31)), (D(2021, 1, 1), D(2024, 7, 31))])
    extra_in = []#'45_tcc', '034_sstk', '168_2d', '136_tcw', '137_tcwv']
    extra_out = extra_in #+ ['179_ttr', '142_lsp', '143_cp', 'logtp', '201_mx2t', '202_mn2t']
    #imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=[], input_levels=levels_medium)
    imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_in, input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_out, input_levels=levels_medium, levels=levels_smol)
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           requested_dates = tdates,
                                           only_at_z = [0],
                                           random_timestep_subset = 4,
                                           ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.decoder import SimpleConvDecoder
    conf = ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        encdec_tr_depth=4,
        #oldpr=True,
        #window_size=(3,7,7),
        #patch_size=(2,8,8),
        simple_decoder_patch_size=(1,8,8),
        latent_size=512,
        #pr_dims = [48, 192, 768],
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
    )
    conf.slow_start = True

    model = ForecastModel(imesh,conf,
                          encoders=[SimpleConvEncoder(imesh,conf)],
                          decoders=[SimpleConvDecoder(omesh,conf)])
    print(model)

    print_total_params(model)

    #config.timesteps = timesteps
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 20_000
    config.latent_l2 = 1e-4
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.optim = 'shampoo'
    #config.shampoo.version = 'new'
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched.schedule_dts = True
    #config.lr_sched.steps_till_max_dt_max = 5_000 
    #config.lr_sched.num_random_subset_max = 5
    config.lr_sched.max_dt_min = 24*3
    config.lr_sched.max_dt_max = 24*14
    config.lr_sched.cosine_period = 5_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.cosine_bottom = 5e-8
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    config.save_imgs_every = 100
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'muir': 6},port=29502, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')
#@launch(ddp=False, kill_nvidia=False)
def Dec25_freyja():
    #config.gpus = '2'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(2008, 1, 1), D(2019, 12, 31)), (D(2021, 1, 1), D(2024, 7, 31))])
    extra = ['45_tcc', '034_sstk', '168_2d', '179_ttr', '136_tcw', '137_tcwv',
             '142_lsp', '143_cp'] #, 'logtp', '201_mx2t', '202_mn2t']
    #imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=[], input_levels=levels_medium)
    imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=[], input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_ecm1)
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           requested_dates = tdates,
                                           only_at_z = [0],
                                           random_timestep_subset = 4,
                                           ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig, StackedConvPlusDecoder
    from model_latlon.encoder import SimpleConvEncoder
    conf = ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        encdec_tr_depth=4,
        #oldpr=True,
        #window_size=(3,7,7),
        #patch_size=(2,8,8),
        latent_size=512,
        pr_dims = [48, 192, 512],
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
    )
    conf.slow_start = True

    model = ForecastModel(imesh,conf,
                          encoders=[SimpleConvEncoder(imesh,conf)],
                          decoders=[StackedConvPlusDecoder(omesh,conf)])
    print(model)

    print_total_params(model)

    #config.timesteps = timesteps
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 20_000
    config.latent_l2 = 1e-4
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.optim = 'shampoo'
    #config.shampoo.version = 'new'
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched.schedule_dts = True
    #config.lr_sched.steps_till_max_dt_max = 5_000 
    #config.lr_sched.num_random_subset_max = 5
    config.lr_sched.max_dt_min = 24*3
    config.lr_sched.max_dt_max = 24*21
    config.lr_sched.cosine_period = 10_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.cosine_bottom = 5e-8
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    config.save_imgs_every = 100
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'singing': 4},port=29502, start_method="spawn", kill_nvidia=False, zulip=True, ping='@**Anuj Shetty**')
#@launch(ddp=False, kill_nvidia=False)
def Dec20_wodan():
    #config.gpus = '2'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(2008, 1, 1), D(2019, 12, 31)), (D(2021, 1, 1), D(2024, 7, 31))])
    extra = ['45_tcc', '034_sstk', '168_2d', '179_ttr', '136_tcw', '137_tcwv']
    mesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_ecm1)
    timesteps = [24*i for i in range(28)]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates,
                                           only_at_z = [0],
                                           random_timestep_subset = 4,
                                           ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig, StackedConvPlusDecoder
    from model_latlon.encoder import SimpleConvEncoder
    conf = ForecastModelConfig(
        [mesh],
        encdec_tr_depth=4,
        oldpr=True,
        latent_size=512,
        pr_dims = [48, 192, 512],
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
    )
    conf.slow_start = True

    model = ForecastModel(mesh,conf,
                          encoders=[SimpleConvEncoder(mesh,conf)],
                          decoders=[StackedConvPlusDecoder(mesh,conf)])
    print(model)

    print_total_params(model)

    #config.timesteps = timesteps
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 20_000
    config.latent_l2 = 1e-4
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.optim = 'shampoo'
    #config.shampoo.version = 'new'
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched.schedule_dts = True
    #config.lr_sched.steps_till_max_dt_max = 5_000 
    #config.lr_sched.num_random_subset_max = 5
    config.lr_sched.max_dt_min = 24*3
    config.lr_sched.max_dt_max = 24*21
    config.lr_sched.cosine_period = 10_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.cosine_bottom = 5e-8
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    config.save_imgs_every = 100
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(nodes={'singing': 4},port=29502, start_method="spawn", kill_nvidia=False, zulip=True, ping='@**Anuj Shetty**')
@launch(ddp=False, kill_nvidia=False)
def Dec11_thor():
    #config.gpus = '2'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    tdates = get_dates([(D(2008, 1, 1), D(2019, 12, 31)), (D(2021, 1, 1), D(2024, 7, 31))])
    extra = []#'45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h'] #msnswrf
    mesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_ecm1)
    timesteps = [24*i for i in range(16)]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24)),
                                           random_timestep_subset = 4,
                                           ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig, ResConvEncoder, ResConvDecoder, SimpleConvEncoder, SimpleConvDecoder
    conf = ForecastModelConfig(
        mesh,
        encdec_tr_depth=4,
        oldpr=True,
        latent_size=1024,
        pr_dims = [48, 192, 512],
        pr_depth = 10,
        tr_embedding='rotary',
        processor_dts = [24],
    )
    conf.slow_start = True

    model = ForecastModel(mesh,conf,
                          encoders=[SimpleConvEncoder(mesh,conf)],
                          decoders=[SimpleConvDecoder(mesh,conf)])
    print(model)

    print_total_params(model)

    #config.timesteps = timesteps
    config.disregard_buffer_checksum = True
    config.auto_loss_consts = True
    config.use_neoloader = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 20_000
    config.optim = 'shampoo'
    #config.shampoo.version = 'new'
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched.schedule_dts = True
    #config.lr_sched.steps_till_max_dt_max = 5_000 
    #config.lr_sched.num_random_subset_max = 5
    config.lr_sched.max_dt_min = 24*2
    config.lr_sched.max_dt_max = 24*15
    config.lr_sched.cosine_period = 10_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.cosine_bottom = 5e-8
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    config.save_imgs_every = 100
    # config.num_workers = 4
    # config.prefetch_factor = 2
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'glass': 6},port=29502, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')
#@launch(ddp=False)
def Nov12_2crass2spurious():
    #config.gpus = '1-3'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1970, 1, 1), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 8, 1))])
    extra_in_out = ['45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h'] #msnswrf
    extra_out_only = [] #'142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_medium) #levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium) #levels_ecm1)
    timesteps = [24*i for i in range(16)]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24)),
                                           random_timestep_subset = 4,
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, #8, 
            dims_per_head=32,
            processor_dt=24,
            neorad=True,
            #window_size=(3,7,7)))
            window_size=(5,7,7)))

    config.timesteps = timesteps
    config.auto_loss_consts = True
    config.use_neoloader = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 20_000
    config.optim = 'shampoo'
    #config.shampoo.version = 'new'
    #config.optim = 'adam'
    config.reset_optimizer = False
    config.lr_sched.schedule_dts = True
    config.lr_sched.steps_till_max_dt_max = 20_000 
    config.lr_sched.max_dt_min = 24*2
    config.lr_sched.max_dt_max = 24*15
    config.lr_sched.cosine_period = 20_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 300
    config.lr_sched.lr = 8e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    config.num_workers = 4
    config.prefetch_factor = 2
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'miramar': 6},port=29502, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')
def Oct23_cas2sandra():
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1970, 1, 1), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 6, 1))])
    extra_in_out = ['45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h']
    extra_out_only = ['201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    timesteps = [24,24*3,24*7,24*15]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24)),
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(4,8,8),
            hidden_dim=768, #512,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, 
            dims_per_head=32,
            processor_dt=24,
            neorad=True,
            window_size=(5,7,7)))

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
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'stinson': 4},port=29501, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')#, kill_nvidia=False)
def Oct22_stale():
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1980, 1, 1), D(1994, 12, 28))]) #, (D(2021, 2, 1), D(2024, 6, 1))])
    load_locs = ['/fast/proc/'] #'/jersey/', 
    mesh = meshes.LatLonGrid(load_locations=load_locs,source='era5-28', input_levels=levels_medium, levels=levels_medium)
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates,
                                           only_at_z = [0,6,12,18],
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=matepoint.checkpoint,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            dims_per_head=32,
            processor_dt=6,
            neorad=True,
            window_size=(5,7,7)))

    config.timesteps = timesteps
    config.auto_loss_consts = True
    config.use_neoloader = False
    config.num_workers = 2
    config.prefetch_factor = 2
    config.pin_memory = True
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 10
    config.timeout = timedelta(minutes=20)
    config.optim = 'shampoo'
    config.shampoo.dim = 2048
    config.reset_optimizer = True
    config.lr_sched.cosine_period = 20_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'stinson': 4},port=29500, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')#, kill_nvidia=False)
def Oct18_fresh():
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(2005, 1, 1), D(2019, 12, 28))]) #, (D(2021, 2, 1), D(2024, 6, 1))])
    load_locs = ['/fast/proc/'] #'/jersey/', 
    mesh = meshes.LatLonGrid(load_locations=load_locs,source='era5-28', input_levels=levels_medium, levels=levels_medium)
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates,
                                           only_at_z = [0,6,12,18],
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=matepoint.checkpoint,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            dims_per_head=32,
            processor_dt=6,
            neorad=True,
            window_size=(5,7,7)))

    config.timesteps = timesteps
    config.auto_loss_consts = True
    config.use_neoloader = False
    config.num_workers = 2
    config.prefetch_factor = 2
    config.pin_memory = True
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 10
    config.timeout = timedelta(minutes=20)
    config.optim = 'shampoo'
    config.shampoo.dim = 2048
    config.reset_optimizer = True
    config.lr_sched.cosine_period = 20_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'bimini': 6},port=29502, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')
def Oct14_naive_s2s_longer():
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1970, 1, 1), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 6, 1))])
    extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_out_only = [] #'142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    timesteps = [24,24*2,24*6,24*12]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24)),
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, 
            dims_per_head=32,
            processor_dt=24,
            neorad=True,
            window_size=(5,7,7)))

    config.timesteps = timesteps
    config.auto_loss_consts = True
    config.use_neoloader = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 10
    config.optim = 'shampoo'
    config.reset_optimizer = False
    config.lr_sched.cosine_period = 25_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 300
    config.lr_sched.lr = 8e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'bimini': 6},port=29501, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')
def Oct10_naive_got_this2s():
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1970, 1, 1), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 6, 1))])
    extra_in_out = ['45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h'] #msnswrf
    extra_out_only = [] #'142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    timesteps = [24,24*2,24*6,24*12]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps,
                                           requested_dates = tdates,
                                           only_at_z = list(range(24)),
                                           ))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, 
            dims_per_head=32,
            processor_dt=24,
            neorad=True,
            window_size=(5,7,7)))

    config.timesteps = timesteps
    config.auto_loss_consts = True
    config.use_neoloader = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.save_every = 100
    config.compute_Bcrit_every = 10
    config.optim = 'shampoo'
    config.reset_optimizer = False
    config.lr_sched.cosine_period = 10_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 300
    config.lr_sched.lr = 8e-4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.save_imgs = True
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

    
@launch(nodes={'ip-172-31-13-240': 8},port=29502, start_method="spawn", zulip=True, ping='@**Anuj Shetty**')
def Sep24_clouddoctorate():
    config.resume = "_"+config.activity.replace("_","-")+"_"
    # config.nope = True
    tdates = get_dates([(D(1977, 5, 1), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    extra = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h'] # last 4 are output only
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25', extra_sfc_vars=extra, bro_zero_me_out=4, input_levels=levels_joank, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25', extra_sfc_vars=extra, input_levels=levels_joank, levels=levels_joank)
    #timesteps = [6,12,24,30,48,96,120] #[6, 12, 24, 72, 144]
    timesteps = [6,24,30,48,96]#,120]
    config.loss_consts = {6: 1.0, 12: 1.0, 24: 1.3, 30: 0.7, 48: 0.4, 96: 0.2, 120: 0.1,  } # <-------- new weights
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, 
                                           max_ram_manual=int(7e9),
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
            #timesteps=timesteps, 
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
    config.DH = 24
    config.validate_N = 8
    config.log_every = 25
    config.save_every = 100
    config.save_imgs = True
    config.optim = 'shampoo'
    config.shampoo.dim = 8192
    config.shampoo.version = 'old'
    #config.optim = 'adam'
    config.reset_optimizer = True
    config.reset_steps_on_resume = False #True
    #config.lr_sched = SimpleNamespace()
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 25_000 # <------- new
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.div_factor= 4
    config.lr_sched.restart_warmup_end_step = 100
    #config.lr_sched.restart_warmup_end_step = 0 # <------- so that we don't do a nested warmup to the warmup rate
    config.lr_sched.lr = 0.05e-3 
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    #config.initial_gradscale = 1024.
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(ddp=0)
#@launch(nodes={'halfmoon': 1},port=29505, start_method="spawn")#, zulip=True, ping='@**John Dean**',validate=False,kill_nvidia=True)
def Sep6_doctorate_test():
    #config.gpus = '0-2'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2014, 12, 28))])#, (D(2021, 2, 1), D(2024, 5, 1))])
    extra_in_out = ['034_sstk']#'15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    extra_out_only = ['142_lsp-6h']#, '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28', extra_sfc_vars=extra_all, output_only_vars=extra_out_only, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28', extra_sfc_vars=extra_all, output_only_vars=extra_out_only, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6] #, 24, 36]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(2e9),
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
            hidden_dim=512, #1408, 
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=3, 
            neorad=True, 
        window_size=(3,5,7)))
    
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
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

@launch(nodes={'ip-172-31-10-41': 8}, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Aug26_cirrus_serpbach3():

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
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True,levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra,is_output_only=True, levels=levels_joank)
    timesteps = [6,24,72,144]
    #timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
        timesteps=timesteps, max_ram_manual=int(100e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
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

    model.do_sub = False
    # config.on_cloud = True
    #config.timeout = timedelta(minutes=3)
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.DH = 24
    config.log_every = 25
    config.save_every = 100
    config.optim = 'shampoo'
    config.reset_optimizer = False
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    # Scaling by sqrt(3) now that we're using 
    config.lr_sched.lr = 0.25e-3
    config.lr_sched.step_offset = 0
    #config.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'ip-172-31-10-41': 8}, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Aug13_cirrus_bachelor():

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
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True,levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra,is_output_only=True, levels=levels_joank)
    timesteps = [6]
    #timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
        timesteps=timesteps, max_ram_manual=int(100e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStep3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8), 
                                                  hidden_dim=32, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, output_deltas=False, neorad=True, window_size=(3,3,3)))

    model.do_sub = False
    # config.on_cloud = True
    #config.timeout = timedelta(minutes=3)
    config.ignore_train_safegaurd = True
    #config.initial_gradscale = 2.
    config.DH = 24
    config.log_every = 25
    config.save_every = 50
    config.optim = 'shampoo'
    config.reset_optimizer = False
    config.lr_sched.cosine_en = True
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000
    # Scaling by sqrt(3) now that we're using 
    config.lr_sched.lr = 0.25e-3 
    config.lr_sched.step_offset = 0
    #config.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    config.dates = tdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'ip-172-31-0-196':4},port=29501, start_method="spawn", kill_nvidia=False)
#@launch(ddp=0,start_method='spawn')
def Aug15_ebs_cirrus():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    #config.gpus = '4-7'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    #config.use_tf32 = True
    #tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    tdates = get_dates([(D(1979, 1, 23), D(1979, 12, 28))]) #, (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, levels=levels_medium)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, levels=levels_medium)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(100e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStep3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(4,8,8), 
                                                  hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))

    model.do_sub = False
    config.HALF = True
    config.on_cloud = False
    config.timeout = timedelta(minutes=3)
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
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000

    config.lr_sched.lr = 0.15e-3 #* 1.4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'ip-172-31-0-196':8},port=29501, start_method="spawn", kill_nvidia=False)
#@launch(ddp=0,start_method='spawn')
def Aug9_hareND():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    #config.gpus = '4-7'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    #config.use_tf32 = True
    #tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    tdates = get_dates([(D(1979, 1, 23), D(1979, 12, 28))]) #, (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True, levels=levels_joank)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(100e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStep3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8), 
                                                  hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, 
                                                  load_half=False, window_size=(3,5,5)))

    model.do_sub = False
    config.HALF = True
    config.on_cloud = False
    config.timeout = timedelta(minutes=3)
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
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.cosine_bottom = 5e-8
    config.lr_sched.warmup_end_step = 1000

    config.lr_sched.lr = 0.15e-3 #* 1.4
    config.adam = SimpleNamespace()
    config.adam.betas = (0.9, 0.99)
    config.adam.weight_decay = 0.001
    config.lr_sched.step_offset = 0
    config.dates = tdates
    config.val_dates = vdates
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'ip-172-31-59-127':8},port=29502, start_method="spawn", kill_nvidia=False)
#@launch(ddp=0,start_method='spawn')
def Aug9_tortoiseND():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    # config.gpus = '0-3'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.resume = "/fast/ignored/runs/run_Jul31-slideux_20240731-173500/model_epoch0_iter112792_step14099_loss0.104.pt"
    #config.new_resume_folder = True
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True, levels=levels_joank)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8), 
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

    w.preserved_conf.lr_sched.lr = 0.15e-3 #* 1.4
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.preserved_conf.on_cloud = True
    w.run()


@launch(nodes={'ip-172-31-59-127':8},port=29502, start_method="spawn", kill_nvidia=False)
def Aug7_cloudserpent():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-7'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra,is_output_only=True, levels=levels_joank)
    timesteps = [6,12,24,48,72]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint, patch_size=(5,6,6), 
                                                  hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
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
    w.preserved_conf.save_every = 10#0
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 100

    w.preserved_conf.lr_sched.lr = 0.25e-3 #* 1.4
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.preserved_conf.on_cloud = True
    w.run()

@launch(nodes={'ip-172-31-59-127':4},port=29502, start_method="spawn", kill_nvidia=False)
def Aug5_slideND():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-3'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra,is_output_only=True, levels=levels_joank)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint, patch_size=(5,6,6), 
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
    w.preserved_conf.save_every = 10#0
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 100

    w.preserved_conf.lr_sched.lr = 0.25e-3 #* 1.4
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'ip-172-31-59-127':4},port=29500, start_method="spawn")
def Aug5_slideD():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '4-7'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra,is_output_only=True, levels=levels_joank)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint, patch_size=(5,6,6), 
                                                  hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
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
    w.preserved_conf.save_every = 10#0
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 100

    w.preserved_conf.lr_sched.lr = 0.25e-3 #* 1.4
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'ip-172-31-59-127':8},port=29502, start_method="spawn")
def Aug2_slide_ebs():
    #import natten
    #assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-7'
    config.use_tf32 = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 1), D(1989, 12, 30))]) #, (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, levels=levels_medium)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, levels=levels_medium)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), 
                                                  hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, 
                                                  load_half=False, window_size=(2,6,12)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 10 #100
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
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
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'ip-172-31-59-127':8},port=29502, start_method="spawn")
def Aug2_steed():
    # import natten
    # assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-7'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(1981, 12, 28))]) #, (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, levels=levels_medium)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, levels=levels_medium)
    timesteps = [24] # [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), window_size=(2,3,6), 
            hidden_dim=640, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, 
            timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=True, output_deltas=True, 
            decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True))
        # data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,6,6), 
        #                                           hidden_dim=640, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
        #                                           processor_dt=3, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,5)))

    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.HALF = False
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.initial_gradscale = 2.
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 50
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
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
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'ip-172-31-59-127':8},port=29502, start_method="spawn")
def Aug1_slidewide():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-7'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True)#, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra,is_output_only=True)#, levels=levels_joank)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,
                                                    patch_size=(4,6,6), 
                                                  hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
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
    w.preserved_conf.save_every = 10#0
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.optim = 'adam'
    w.preserved_conf.reset_optimizer = False
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
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'ip-172-31-59-127':8},port=29502, start_method="spawn")
def Jul31_slideux():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-7'
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra,is_output_only=True, levels=levels_joank)
    timesteps = [6,24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(5,4,4), 
                                                  hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
                                                  processor_dt=3, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, 
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

    w.preserved_conf.lr_sched.lr = 0.25e-3 #* 1.4
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'ip-172-31-59-127':8},port=29502, start_method="spawn")
#@launch(ddp=0,start_method='spawn')
def Jul30_slidetest():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '0-7'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(1979, 1, 23), D(1979, 3, 30))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    #omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)

    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra, is_output_only=True, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-25',extra_sfc_vars=extra,is_output_only=True, levels=levels_joank)
    timesteps = [6,24]
    #timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
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
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1000

    w.preserved_conf.lr_sched.lr = 0.3e-3 #* 1.4
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(nodes={'ip-54-190-147-2243':2},port=29501,start_method='spawn')
#@launch(ddp=0,start_method='spawn')
def Jul29_goku():
    import natten
    assert natten.DO_WRAP == True

    #### NATTEN COMPILED WITH WRAP
    #### RPB AND EPB turned on in model_latlon_3d.py!!

    config.gpus = '6-7'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    #tdates = get_dates([(D(2022, 1, 1), D(2021, 1, 2))])
    vdates = get_dates((D(2021, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=2, CLOUD=0)#, levels=levels_ecm2)
    #extra = ['logtp', '15_msnswrf', '45_tcc']
    extra = []
    #mesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    #imesh = meshes.LatLonGrid(WEATHERBENCH=2, CLOUD=0,source='era5-28',extra_sfc_pad=3, levels=levels_gfs)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_medium)
    timesteps = [6,24]
    #timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates
                                           ))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, Transformer=SlideLayer3D, checkpointfn=matepoint.checkpoint,patch_size=(4,6,6), 
                                                  hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32, 
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
    w.preserved_conf.reset_optimizer = False
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

#@launch(nodes={'ip-172-31-59-127':2},start_method='spawn')
@launch(ddp=0,start_method='spawn')
def Jul29_goku_0():
    config.gpus = '7' #0,1,2,3,5
    # config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    # config.loss_consts[24] = 1
    # config.loss_consts[48] = 0.25
    # config.coupled.hell = True
    # config.coupled.B = 768
    # config.coupled.config.weights = np.array([3, 3, 3, 4, 4, 0.1, 0.1], dtype=np.float32)
    # config.coupled.config.weight = 0.5
    # config.coupled.model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True)
    # check = torch.load('/fast/model_step71500_loss0.135.pt', map_location='cpu')
    # config.coupled.model.load_state_dict(check['model_state_dict'],strict=True)
    # for param in config.coupled.model.parameters():
    #     param.requires_grad = False
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra = []
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    # imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=1,source='era5-28',extra_sfc_vars=extra,is_output_only=True)

    timesteps = [6,24]
    # timesteps = [6,24,72]
    # timesteps = [24, 48]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[imesh], outputs=[omesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = tdates,
                                           #only_at_z=[0,6,12,18]
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outAputs=data.config.outputs, checkpointfn=matepoint.checkpoint,patch_size=(4,6,6),
                                                  hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32,
                                                  processor_dt=6, use_matepoint=True, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(4,3,6)))
    # model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,4,4), hidden_dim=1024, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=10, 
    #                                                 lat_compress=False, timesteps=[24], train_timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=False, output_deltas=True, 
    #                                                 decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(4,3,6)))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], train_timesteps=[HH], dims_per_head=32, processor_dt=S0, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=True))
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.ignore_train_safegaurd = True
    w.preserved_conf.validate_every = -1
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 25
    w.preserved_conf.save_every = 250
    w.preserved_conf.optim = 'adam'
    #w.preserved_conf.optim = 'shampoo'
    w.preserved_conf.diff_loss = 5.
    w.preserved_conf.batch_size = 1
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_en = True
    w.preserved_conf.lr_sched.cosine_period = 45_000
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
    
@launch(nodes={'miramar':5},port=29500, start_method="spawn",ddp=1)
def Jul16_steed_1():
    config.gpus = '0-5' #0,1,2,3,5
    config.resume = "_"+config.activity.replace("_","-")+"_"
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra, is_output_only=True,levels=levels_medium)


    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],timesteps=timesteps,worker_complain = False,requested_dates = tdates,max_ram_manual = int(6e9)))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(
        ForecastStepConfig(
            data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), window_size=(2,3,6), 
            hidden_dim=1024, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, 
            timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=True, 
            decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True))
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

@launch(nodes={'stinson':4},port=29500, start_method="spawn",ddp=1)
def Jul16_steed():
    #config.gpus = '0-5' #0,1,2,3,5
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    config.resume = '/fast/ignored/runs/run_Jul16-steed_20240717-162718/model_epoch0_iter7996_step1999_loss0.060.pt'
    #config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra, is_output_only=True,levels=levels_medium)


    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],timesteps=timesteps,worker_complain = False,requested_dates = tdates,max_ram_manual = int(4e9)))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(
        ForecastStepConfig(
            data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), window_size=(2,3,6), 
            hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, 
            timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=True, 
            decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True))
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
    w.preserved_conf.optim = 'shampoo'
    #w.preserved_conf.reset_optimizer = True
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

if __name__ == '__main__':
    run(locals().values())
