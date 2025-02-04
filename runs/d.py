from launch import *

if am_i_torchrun():
    from train import *

from train import * # <--- comment this back in if not using DDP
from torch.utils import checkpoint as torch_checkpoint

#@launch(ddp=0,start_method='spawn')
@launch(nodes={'stinson': 2} ,port=29524, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Dec3_regionaldecoder_test():
    #config.prefix = '_penguins'
    config.nope = False
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvDecoder, RegionalSimpleConvDecoder
    conf = ForecastModelConfig(
        mesh,
        encdec_tr_depth=4,
        oldenc=True,
        latent_size=896,
        window_size=(3,5,7),
        decoder_types=[SimpleConvDecoder, RegionalSimpleConvDecoder],
        decoder_configs=[
            SimpleNamespace(), 
            SimpleNamespace(
                latent_bounds = (14,34,113,149),
                real_bounds = (112,272,904,1192)
            )]
    )
    model = ForecastModel(mesh,conf)
    print(model)

    print_total_params(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.shampoo.version = 'old'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 22_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1,24:1}
    config.decoder_loss_weights = [1, 0.5]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0,start_method='spawn')
@launch(nodes={'muir': 2} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Nov26_conusdecoder_test():
    #config.prefix = '_penguins'
    config.nope = False
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(mesh,encdec_tr_depth=4,oldenc=True,conusdec=True,olddec=True,latent_size=896,window_size=(3,5,7))
    model = ForecastModel(mesh,conf)
    print(model)

    print_total_params(model)

    # from evals.package_neo import get_serp3bachelor
    # bachelor = get_serp3bachelor()
    # print(bachelor)
    # print_total_params(bachelor)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.shampoo.version = 'old'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 22_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    #config.save_imgs_every = 100
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1,24:1}
    config.decoder_loss_weights = [1, 0.5]
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()
    
@launch(nodes={'muir': 2} ,port=29505, start_method="spawn",clear_cache=False, kill_nvidia=False, zulip=True, ping='@**John Dean**')
#@launch(ddp=0,start_method='spawn')
def Dec3_deltas_200k_2g_normed_ib_offset():
    config.prefix = '_diffusion3'
    config.nope = 0
    config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from evals.package_neo import get_serp3bachelor
    bachelor = get_serp3bachelor()
    # turn off gradients for bachelor
    for p in bachelor.parameters():
        p.requires_grad = False

    from diffusion.model import UNet

    from model_latlon.diffusion import ForecastCombinedDiffusion


    diffuser = UNet(
        in_channels=mesh.n_sfc_vars*2,
        out_channels=mesh.n_sfc_vars,
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        conditioning_channels=512
    )

    print(diffuser)

    model = ForecastCombinedDiffusion(forecaster=bachelor,diffuser=diffuser, schedule='linear',append_input=True)

    config.diffusion = True
    config.bboxes = [[0,1/2,0,1],[1/2,1,0,1]]
    #config.bboxes = [[0,1/3,0,1],[1/3,2/3,0,1],[2/3,1,0,1]]
    #config.bboxes = [[0,1/3,0,1/3],[1/3,2/3,0,1/3],[2/3,1,0,1/3],[0,1/3,1/3,2/3],[1/3,2/3,1/3,2/3],[2/3,1,1/3,2/3],[0,1/3,2/3,1],[1/3,2/3,2/3,1],[2/3,1,2/3,1]]
    config.rerun_sample = len(config.bboxes) * len(timesteps) * 5 
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 500
    config.HALF = True
    config.optim = 'shampoo'
    config.optim = 'adam'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 200_000
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.lr_sched.restart_warmup_end_step = 0
    config.save_imgs_every = 100000000
    config.strict_load = False
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 2} ,port=29504, start_method="spawn",clear_cache=False, kill_nvidia=False, zulip=True, ping='@**John Dean**')
#@launch(ddp=0,start_method='spawn')
def Nov30_deltas_100k_2g_normed():
    config.prefix = '_diffusion3'
    config.nope = 0
    config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from evals.package_neo import get_serp3bachelor
    bachelor = get_serp3bachelor()
    print(bachelor)
    # turn off gradients for bachelor
    for p in bachelor.parameters():
        p.requires_grad = False

    from diffusion.model import UNet

    from model_latlon.diffusion import ForecastCombinedDiffusion


    diffuser = UNet(
        in_channels=mesh.n_sfc_vars,
        out_channels=mesh.n_sfc_vars,
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        conditioning_channels=512
    )

    model = ForecastCombinedDiffusion(forecaster=bachelor,diffuser=diffuser, schedule='linear')

    config.diffusion = True
    config.bboxes = [[0,1/2,0,1],[1/2,1,0,1]]
    #config.bboxes = [[0,1/3,0,1],[1/3,2/3,0,1],[2/3,1,0,1]]
    #config.bboxes = [[0,1/3,0,1/3],[1/3,2/3,0,1/3],[2/3,1,0,1/3],[0,1/3,1/3,2/3],[1/3,2/3,1/3,2/3],[2/3,1,1/3,2/3],[0,1/3,2/3,1],[1/3,2/3,2/3,1],[2/3,1,2/3,1]]
    config.rerun_sample = len(config.bboxes) * len(timesteps) * 5 
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 500
    config.HALF = True
    config.optim = 'shampoo'
    config.optim = 'adam'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 100_000
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.lr_sched.restart_warmup_end_step = 0
    config.save_imgs_every = 100000000
    config.strict_load = False
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

# real run is on /huge/users/djoan10. copied here with the new changes for convenience
@launch(nodes={'barceloneta': 3} ,port=29506, start_method="spawn",clear_cache=False, kill_nvidia=False)#,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Nov29_rotary():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk',
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    #imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    #omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    timesteps = [0,6,24,48]#,36]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = True
    conf.slow_start = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1280
    #conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    conf.affine = True
    conf.pr_depth = 10
    conf.encdec_tr_depth = 4
    conf.oldpr = True
    conf.dims_per_head = 40 # (1280/40 = 32 heads)
    conf.tr_embedding = 'rotary'
    model = ForecastModel(imesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 2048
    #config.shampoo.dim = 1024
    config.reset_optimizer = False
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    #config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'muir': 3} ,port=29504, start_method="spawn",clear_cache=False, kill_nvidia=False, zulip=True, ping='@**John Dean**')
#@launch(ddp=0,start_method='spawn')
def Nov29_mutlitimestep():
    config.prefix = '_diffusion3'
    config.nope = 0
    config.gpus = '0-2'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,24,36,48,72]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from evals.package_neo import get_serp3bachelor
    bachelor = get_serp3bachelor()
    print(bachelor)
    # turn off gradients for bachelor
    for p in bachelor.parameters():
        p.requires_grad = False

    from diffusion.model import UNet

    from model_latlon.diffusion import ForecastCombinedDiffusion

    diffuser = UNet(
        in_channels=mesh.n_sfc_vars,
        out_channels=mesh.n_sfc_vars,
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        conditioning_channels=512
    )

    model = ForecastCombinedDiffusion(forecaster=bachelor,diffuser=diffuser, schedule='linear')

    config.diffusion = True
    config.bboxes = [[0,1/2,0,1],[1/2,1,0,1]]
    #config.bboxes = [[0,1/3,0,1],[1/3,2/3,0,1],[2/3,1,0,1]]
    #config.bboxes = [[0,1/3,0,1/3],[1/3,2/3,0,1/3],[2/3,1,0,1/3],[0,1/3,1/3,2/3],[1/3,2/3,1/3,2/3],[2/3,1,1/3,2/3],[0,1/3,2/3,1],[1/3,2/3,2/3,1],[2/3,1,2/3,1]]
    config.rerun_sample = len(config.bboxes) * len(timesteps)
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 500
    config.HALF = True
    config.optim = 'shampoo'
    config.optim = 'adam'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 200_000
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.lr_sched.restart_warmup_end_step = 0
    config.save_imgs_every = 100000000
    config.strict_load = False
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 1} ,port=29502, start_method="spawn",clear_cache=False, kill_nvidia=False, zulip=True, ping='@**John Dean**')
#@launch(ddp=0,start_method='spawn')
def Nov28_cond512():
    config.prefix = '_diffusion3'
    config.nope = 0
    config.gpus = '0'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from evals.package_neo import get_serp3bachelor
    bachelor = get_serp3bachelor()
    print(bachelor)
    # turn off gradients for bachelor
    for p in bachelor.parameters():
        p.requires_grad = False

    from diffusion.model import UNet

    from model_latlon.diffusion import ForecastCombinedDiffusion

    diffuser = UNet(
        in_channels=mesh.n_sfc_vars,
        out_channels=mesh.n_sfc_vars,
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        conditioning_channels=512
    )

    model = ForecastCombinedDiffusion(forecaster=bachelor,diffuser=diffuser, schedule='linear')

    config.diffusion = True
    config.bboxes = [[0,1/2,0,1],[1/2,1,0,1]]
    #config.bboxes = [[0,1/3,0,1],[1/3,2/3,0,1],[2/3,1,0,1]]
    #config.bboxes = [[0,1/3,0,1/3],[1/3,2/3,0,1/3],[2/3,1,0,1/3],[0,1/3,1/3,2/3],[1/3,2/3,1/3,2/3],[2/3,1,1/3,2/3],[0,1/3,2/3,1],[1/3,2/3,2/3,1],[2/3,1,2/3,1]]
    config.rerun_sample = len(config.bboxes) * 2
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 500
    config.HALF = True
    config.optim = 'shampoo'
    config.optim = 'adam'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 200_000
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.lr_sched.restart_warmup_end_step = 0
    config.save_imgs_every = 100000000
    config.strict_load = False
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'muir': 1} ,port=29502, start_method="spawn",clear_cache=False, kill_nvidia=False, zulip=True, ping='@**John Dean**')
#@launch(ddp=0,start_method='spawn')
def Nov23_224_24hr_linear_400k():
    config.prefix = '_diffusion3'
    config.nope = 0
    config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from evals.package_neo import get_serp3bachelor
    bachelor = get_serp3bachelor()
    print(bachelor)
    # turn off gradients for bachelor
    for p in bachelor.parameters():
        p.requires_grad = False

    from diffusion.model import UNet

    from model_latlon.top import ForecastCombinedDiffusion

    diffuser = UNet(
        in_channels=mesh.n_sfc_vars,
        out_channels=mesh.n_sfc_vars,
        base_channels=224,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256
    )

    model = ForecastCombinedDiffusion(forecaster=bachelor,diffuser=diffuser, schedule='linear')

    config.diffusion = True
    #config.bboxes = [[0,1/2,0,1],[1/2,1,0,1]]
    #config.bboxes = [[0,1/3,0,1],[1/3,2/3,0,1],[2/3,1,0,1]]
    config.bboxes = [[0,1/3,0,1/3],[1/3,2/3,0,1/3],[2/3,1,0,1/3],[0,1/3,1/3,2/3],[1/3,2/3,1/3,2/3],[2/3,1,1/3,2/3],[0,1/3,2/3,1],[1/3,2/3,2/3,1],[2/3,1,2/3,1]]
    config.rerun_sample = len(config.bboxes) * 2
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 500
    config.HALF = True
    config.optim = 'shampoo'
    config.optim = 'adam'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 400_000
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.lr_sched.restart_warmup_end_step = 0
    config.save_imgs_every = 100000000
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'muir': 1} ,port=29508, start_method="spawn",clear_cache=False, kill_nvidia=False, zulip=True, ping='@**John Dean**')
#@launch(ddp=0,start_method='spawn')
def Nov23_128_24hr_linear_400k():
    config.prefix = '_diffusion3'
    config.nope = 0
    config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from evals.package_neo import get_serp3bachelor
    bachelor = get_serp3bachelor()
    print(bachelor)
    # turn off gradients for bachelor
    for p in bachelor.parameters():
        p.requires_grad = False

    from diffusion.model import UNet

    from model_latlon.top import ForecastCombinedDiffusion

    diffuser = UNet(
        in_channels=mesh.n_sfc_vars,
        out_channels=mesh.n_sfc_vars,
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256
    )

    model = ForecastCombinedDiffusion(forecaster=bachelor,diffuser=diffuser, schedule='linear')

    config.diffusion = True
    #config.bboxes = [[0,1/2,0,1],[1/2,1,0,1]]
    config.bboxes = [[0,1/3,0,1],[1/3,2/3,0,1],[2/3,1,0,1]]
    config.rerun_sample = len(config.bboxes) * 5 
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 500
    config.HALF = True
    config.optim = 'shampoo'
    config.optim = 'adam'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 400_000
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.lr_sched.restart_warmup_end_step = 0
    config.save_imgs_every = 100000000
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 1} ,port=29507, start_method="spawn",clear_cache=False, kill_nvidia=False, zulip=True, ping='@**John Dean**')
#@launch(ddp=0,start_method='spawn')
def Nov23_96_24hr_linear_200k_1e4():
    config.prefix = '_diffusion3'
    config.nope = 0
    config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from evals.package_neo import get_serp3bachelor
    bachelor = get_serp3bachelor()
    print(bachelor)
    # turn off gradients for bachelor
    for p in bachelor.parameters():
        p.requires_grad = False

    from diffusion.model import UNet

    from model_latlon.top import ForecastCombinedDiffusion

    diffuser = UNet(
        in_channels=mesh.n_sfc_vars,
        out_channels=mesh.n_sfc_vars,
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256
    )

    model = ForecastCombinedDiffusion(forecaster=bachelor,diffuser=diffuser, schedule='linear')

    config.diffusion = True
    config.bboxes = [[0,1/2,0,1],[1/2,1,0,1]]
    config.rerun_sample = len(config.bboxes) * 5 
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 500
    config.HALF = True
    config.optim = 'shampoo'
    config.optim = 'adam'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 200_000
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.lr_sched.restart_warmup_end_step = 0
    config.save_imgs_every = 100000000
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'gold': 3} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**John Dean**')
#@launch(ddp=0,start_method='spawn')
def Nov15_bachelorclone_gen2():
    #config.prefix = '_penguins'
    config.nope = False
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)

    timesteps = [6,24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    model = ForecastStep3D(
    ForecastStepConfig(
        [mesh],
        outputs=[mesh],
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
        neorad=True,
        window_size=(3,5,7)))
    print(model)
    print_total_params(model)

    from evals.package_neo import get_serp3bachelor
    bachelor = get_serp3bachelor()
    print(bachelor)
    print_total_params(bachelor)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.shampoo.version = 'notold'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1,24:1}
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'gold': 3} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**John Dean**')
#@launch(ddp=0,start_method='spawn')
def Nov15_bachelorclone_nocontorsion():
    #config.prefix = '_penguins'
    config.nope = False
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(mesh,encdec_tr_depth=4,oldenc=True,olddec=True,latent_size=896,window_size=(3,5,7))
    model = ForecastModel(mesh,conf)
    print(model)

    print_total_params(model)

    from evals.package_neo import get_serp3bachelor
    bachelor = get_serp3bachelor()
    print(bachelor)
    print_total_params(bachelor)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.shampoo.version = 'notold'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1,24:1}
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'miramar': 6} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Nov12_oldpr():
    #config.prefix = '_penguins'
    config.nope = True
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0,6,24,48]#,36]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1024
    conf.affine = True
    conf.oldenc = False
    conf.oldpr = True
    model = ForecastModel(imesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    #config.disregard_buffer_checksum = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.reset_optimizer = False
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    #config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 40_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'bimini': 6} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Nov7_smallvertconv_4tran():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    timesteps = [0,6,24,48]#,36]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1280
    conf.pr_dims = [48, 192, 640]
    conf.affine = True
    conf.encdec_tr_depth = 4
    model = ForecastModel(imesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.reset_optimizer = False
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    #config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'singing': 4, 'baga': 4} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Nov7_smallvertconv():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    timesteps = [0,6,24,48]#,36]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1280
    conf.pr_dims = [48, 192, 640]
    conf.affine = True
    model = ForecastModel(imesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.reset_optimizer = False
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    #config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'miramar': 6} ,port=29503, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Nov6_oldenc():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0,6,24,48]#,36]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1024
    conf.affine = True
    conf.oldenc = True
    model = ForecastModel(imesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.compute_Bcrit_every = np.nan
    config.reset_optimizer = False
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    #config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 40_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 60
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'miramar': 6} ,port=29503, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Nov2_jal_noaffine():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0,6,24,48]#,36]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1024
    conf.affine = False
    model = ForecastModel(imesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.compute_Bcrit_every = np.nan
    config.reset_optimizer = False
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    #config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 40_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 60
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



#@launch(ddp=0,start_method='spawn')
@launch(nodes={'stinson': 4} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=True)#, zulip=True, ping='@**John Dean**',validate=False)
def Oct27_bachelor_haretest():
    #config.prefix = '_penguins'
    config.nope = True
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    timesteps = [0,6,24]#,36]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(mesh)
    conf.latent_size = 1280
    #conf.harebrained = True
    model = ForecastModel(mesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 2048
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 22_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'glass': 6} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Nov1_joan_always_loses():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0,6,24,48]#,36]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1024
    model = ForecastModel(imesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.compute_Bcrit_every = np.nan
    config.reset_optimizer = False
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    #config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 40_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 60
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()




@launch(nodes={'barceloneta': 6} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Oct31_joan_never_wins():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra_input = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0,6,24,48]#,36]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1024
    conf.affine = True
    conf.oldenc = False
    model = ForecastModel(imesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    #config.disregard_buffer_checksum = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.reset_optimizer = False
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    #config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 40_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 60
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'muir': 6} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=True)#, zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Oct27_bachelorlike():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    timesteps = [0,6,24]#,36]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(mesh)
    conf.latent_size = 1024
    model = ForecastModel(mesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()
@launch(nodes={'bimini': 3} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def Oct25_resconv_24():
    #config.prefix = '_penguins'
    #config.nope = True
    config.gpus = '3,4,5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    timesteps = [24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(mesh)
    model = ForecastModel(mesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 45_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'bimini': 2} ,port=29504, start_method="spawn",clear_cache=False)#,kill_nvidia=False, zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def chillout():
    config.prefix = '_penguins'
    #config.nope = True
    config.gpus = '0'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    timesteps = [0]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(mesh)
    model = ForecastModel(mesh,conf)
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 25_000
    config.lr_sched.warmup_end_step = 400
    config.lr_sched.lr = 1e-3
    config.lr_sched.restart_warmup_end_step = 500
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'glass': 1} ,port=29504, start_method="spawn",clear_cache=False)#,kill_nvidia=False, zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def hareC():
    config.prefix = '_diffusion'
    #config.nope = True
    config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [0]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from diffusion.model import UNet, get_timestep_embedding
    from evals.package_neo import get_serp3bachelor


    model = SfcOnlyHareEncoder(mesh)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 2048
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 25_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 0.1e-3
    config.lr_sched.restart_warmup_end_step = 500
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


@launch(nodes={'glass': 1} ,port=29504, start_method="spawn",clear_cache=False)#,kill_nvidia=False, zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def hare_autoencoder_reflect():
    config.prefix = '_diffusion'
    #config.nope = True
    config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [0]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from diffusion.model import UNet, get_timestep_embedding
    from evals.package_neo import get_serp3bachelor


    model = SfcOnlyAutoencoder(mesh)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 2048
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 25_000
    config.lr_sched.warmup_end_step = 400
    config.lr_sched.lr = 1e-3
    config.lr_sched.restart_warmup_end_step = 500
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'glass': 3} ,port=29504, start_method="spawn")#,kill_nvidia=False, zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def diffusion():
    config.prefix = '_diffusion'
    #config.nope = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [6]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from diffusion.model import UNet, get_timestep_embedding
    from evals.package_neo import get_serp3bachelor

    # Model, optimizer, and loss function
    forecaster = get_serp3bachelor()
    forecaster.eval()
    del forecaster.decoders
    forecaster.decoders = None
    for param in forecaster.parameters():
        param.requires_grad = False


    diffuser = UNet(
        in_channels=mesh.n_sfc_vars,
        out_channels=mesh.n_sfc_vars,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256
    )

    model = ForecastStepDiffusion(forecaster=forecaster,diffuser=diffuser)

    config.diffusion = True
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.HALF = True
    config.optim = 'shampoo'
    config.optim = 'adam'
    config.shampoo.dim = 8192
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 5_000
    config.lr_sched.warmup_end_step = 25
    config.lr_sched.lr = 0.6e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.save_imgs_every = 100000000
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()





if __name__ == '__main__':
    run(locals().values())
