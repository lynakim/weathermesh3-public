import sys 
sys.path.append('.') # what the fuck
from runs.launch import *

if am_i_torchrun():
    from train.trainer import *

from train.trainer import * # <--- comment this back in if not using DDP


@launch(nodes={'bimini': 4} ,port=29524, start_method="spawn", clear_cache=False, kill_nvidia=False)
#@launch(ddp=0)
def Feb1_john_pointy_regioned():
    config.nope = 0
    config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.point_dec import JohnPointDecoder    
    #from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = JohnPointDecoder(
        station_mesh,
        conf,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [0,6]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=6)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))
    #config.num_workers = 0
    #config.prefetch_factor = None

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 5
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo' #'adam'
    config.shampoo.dim = 4096
    config.shampoo.version = 'old'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched = LRScheduleConfig(
        schedule_dts=False,
        cosine_period=200_000,
        warmup_end_step=1_000,
        restart_warmup_end_step=500,
        lr=0.7e-4,
    )
    config.weight_eps = 0.02
    w = WeatherTrainer(conf=config,model=model,data=data)
    #return w
    w.run()

#@launch(nodes={'barceloneta': 6} ,port=29506, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
@launch(ddp=0)
def Jan12_daforecast_js():
    config.nope = 1
    tdates = get_dates([(D(1979, 1, 23), D(2025, 1, 1))])

    from evals.package_neo import get_joanrealtimesucks_hresonly
    joansucks = get_joanrealtimesucks_hresonly()
    imesh = joansucks.encoders[0].mesh
    imesh.hour_offset = 6
    omesh = joansucks.decoders[0].mesh

    timesteps = [0,24]

    from model_latlon.da_encoders import MicrowaveData, BalloonData
    mesh_1bamua = MicrowaveData("1bamua") # atms,1bamua
    mesh_atms = MicrowaveData("atms") # atms,1bamua
    from datasets import AnalysisDataset, IgraDataset, MicrowaveDataset
    igra_mesh = BalloonData()
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
            MicrowaveDataset(mesh_1bamua, is_required=False), 
            MicrowaveDataset(mesh_atms, is_required=False), 
            IgraDataset(igra_mesh, is_required=True), 
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        timesteps=timesteps,
        only_at_z=[0,6,12,18],
        requested_dates = tdates
        ))

    config.num_workers = 0
    config.prefetch_factor = None
    from model_latlon.decoder import SimpleConvDecoder
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.da_transformer import DAOnlyModel, DATransformer, DAForecastModel2
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    for param in joansucks.parameters():
        param.requires_grad = False

    conf = joansucks.config
    model = DAForecastModel2(
        conf,
        anl_encoders=[joansucks.encoders[0]],
        obs_encoders=[PointObPrEncoder(mesh_1bamua, conf), PointObPrEncoder(mesh_atms, conf), PointObPrEncoder(igra_mesh, conf)],
        da_transformer=DATransformer(conf),
        decoders=[joansucks.decoders[0]],
        processors={'6':joansucks.processors['6']}
        )
    
    print(model)
    print_total_params(model)

    config.do_da_compare = True
    config.strict_load = True
    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 2
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 512
    config.reset_optimizer = True
    config.reset_steps_on_resume = False#True#False#True
    config.save_optimizer = True
    config.lr_sched.cosine_period = 20_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-4
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 200
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    #return w
    w.run()

#@launch(nodes={'muir': 6} ,port=29506, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
@launch(ddp=0)
def Jan7_daforecast():
    config.nope = 1
    tdates = get_dates([(D(1979, 1, 23), D(2025, 1, 1))])
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    imesh = meshes.LatLonGrid(source='neohres-20',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_hres, levels=levels_medium, hour_offset=6)

    timesteps = [6,30]

    from model_latlon.da_encoders import MicrowaveData, BalloonData
    mesh_1bamua = MicrowaveData("1bamua") # atms,1bamua
    mesh_atms = MicrowaveData("atms") # atms,1bamua
    from datasets import AnalysisDataset, IgraDataset, MicrowaveDataset
    igra_mesh = BalloonData()
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
            MicrowaveDataset(mesh_1bamua, is_required=True), 
            MicrowaveDataset(mesh_atms, is_required=True), 
            IgraDataset(igra_mesh, is_required=True), 
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        timesteps=timesteps,
        only_at_z=[0,6,12,18],
        requested_dates = tdates
        ))

    #config.num_workers = 0
    #config.prefetch_factor = None
    
    conf = ForecastModelConfig(
        None,
        outputs=[omesh],
        #checkpoint_type="torch",
        encdec_tr_depth=4,
        latent_size=896,
        window_size=(3,5,7),
        weight_eps=0.01,
    )

    from model_latlon.decoder import SimpleConvDecoder
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.da_transformer import DAOnlyModel, DATransformer, DAForecastModel2
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    model = DAForecastModel2(
        conf,
        anl_encoders=[SimpleConvEncoder(imesh,conf)],
        obs_encoders=[MicrowaveEncoder(mesh_1bamua, conf), MicrowaveEncoder(mesh_atms, conf), PointObPrEncoder(igra_mesh, conf)],
        da_transformer= DATransformer(conf),
        decoders=[SimpleConvDecoder(omesh, conf)]
        )
    print(model)
    print_total_params(model)

    config.strict_load = True
    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 1024
    config.reset_optimizer = True
    config.reset_steps_on_resume = False#True#False#True
    config.save_optimizer = True
    config.lr_sched.cosine_period = 20_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-4
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 200
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    #return w
    w.run()




@launch(nodes={'muir': 3} ,port=29501, start_method="spawn",clear_cache=False, zulip=True, ping='@**John Dean**')
#@launch(ddp=0)
def Jan4_igraonly():
    config.nope = 0
    tdates = get_dates([(D(1979, 1, 23), D(2025, 1, 1))])
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0]

    from model_latlon.da_encoders import BalloonData
    from datasets import AnalysisDataset, IgraDataset
    igra_mesh = BalloonData(da_timestep_offset=0)
    
    data = WeatherDataset(DataConfig(
        inputs=[
            IgraDataset(igra_mesh, is_required=True), 
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        timesteps=timesteps,
        only_at_z=[0,12],
        requested_dates = tdates
        ))

    #config.num_workers = 0
    #config.prefetch_factor = None
    
    conf = ForecastModelConfig(
        None,
        outputs=[omesh],
        #checkpoint_type="torch",
        encdec_tr_depth=4,
        latent_size=896,
        window_size=(3,5,7),
        weight_eps=0.01,
    )

    from model_latlon.decoder import SimpleConvDecoder
    from model_latlon.da_transformer import DAOnlyModel, DATransformer
    from model_latlon.da_encoders import PointObPrEncoder

    model = DAOnlyModel(
        conf,
        anl_encoders=[],
        obs_encoders=[PointObPrEncoder(igra_mesh, conf)],
        da_transformer= DATransformer(conf),
        decoders=[SimpleConvDecoder(omesh, conf)]
        )
    print(model)
    print_total_params(model)

    config.strict_load = True
    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 1000
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.reset_optimizer = True
    config.reset_steps_on_resume = False#True#False#True
    config.save_optimizer = True
    config.lr_sched.cosine_period = 20_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-4
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 200
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    #return w
    w.run()



@launch(nodes={'glass': 3} ,port=29501, start_method="spawn",clear_cache=False)#,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
#@launch(ddp=0)
def Jan3_datest_shamp():
    #config.prefix = '_penguins'
    config.nope = False
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 2, 21))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh1 = meshes.LatLonGrid(source='neogfs-25',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_gfs, levels=levels_medium)
    imesh2 = meshes.LatLonGrid(source='neohres-20',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_hres, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0,6,24,48]#,36]
    timesteps = [0,6,12,18,24,30,36,42,48,60,72,78,96,120]
    timesteps = [1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,19,20,21,22,23]
    timesteps = [6]
    #timesteps = [6]

    from model_latlon.da_encoders import MicrowaveData, BalloonData
    from datasets import MicrowaveDataset, AnalysisDataset
    mesh_1bamua = MicrowaveData("1bamua") # atms,1bamua
    #igra_mesh = BalloonData()

    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh1), 
            AnalysisDataset(imesh2), 
            MicrowaveDataset(mesh_1bamua),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        timesteps=timesteps,
        only_at_z=[0,6,12,18],
        requested_dates = tdates
        ))

    #config.num_workers = 0
    #config.prefetch_factor = None
    config.lr_sched.schedule_dts = False

    import evals.package_neo as pn

    forecast_model = pn.get_joanrealtimesucks()
    for param in forecast_model.parameters():
        param.requires_grad = False

    from model_latlon.da_transformer import DAForecastModel, DATransformer
    from model_latlon.config import ForecastModelConfig
    from model_latlon.da_encoders import MicrowaveEncoder

    da_config = ForecastModelConfig([imesh1, imesh2, mesh_1bamua])

    encoder_1bamua = MicrowaveEncoder(mesh_1bamua, da_config)

    da_transformer = DATransformer(da_config)

    overall_model = DAForecastModel(forecast_model, [encoder_1bamua], da_transformer)

    config.strict_load = True
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
    config.reset_steps_on_resume = False#True#False#True
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    #config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 20_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.04e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 1000_000
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=overall_model,data=data)
    #return w
    w.run()


@launch(ddp=0,start_method='spawn')
def Dec8_rotary_transplant():
    #config.prefix = '_penguins'
    config.nope = True
    config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"
    from evals.package_neo import get_rotary
    import model_latlon.top as top
    
    if 0:
        model = get_rotary()
    else:
        extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
        extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
                '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

        imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
        omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)

        from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvEncoder, SimpleConvDecoder, ResConvEncoder, ResConvDecoder
        conf = ForecastModelConfig(
            [imesh],
            outputs=[omesh],
            latent_size = 1280,
            pr_dims = [48, 192, 512],
            affine = True,
            pr_depth = 10,
            encdec_tr_depth = 4,
            oldpr = True,
            tr_embedding = 'rotary',
        )
        encoder = ResConvEncoder(imesh,conf)
        decoder = ResConvDecoder(omesh,conf)
        model = ForecastModel(imesh,conf,encoders=[encoder],decoders=[decoder])


    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    for p in model.parameters():
        p.requires_grad = False
    
    decoder = top.StackedConvPlusDecoder(omesh,model.config)
    del model.decoders
    model.decoders = nn.ModuleList([decoder])
    
    print(model)
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=[0],
                                            requested_dates = tdates
                                            ))
 
    config.lr_sched = LRScheduleConfig(
        schedule_dts=True,
        cosine_period=10_000,
        warmup_end_step=500,
        lr=0.3e-3,
        max_dt_min=24,
        max_dt_max=72,
        steps_till_max_dt_max=9_000,
        num_random_subset_min=4,
        num_random_subset_max=4,
    )
    config.lr_sched.make_plots()
    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    #config.shampoo.dim = 1024
    config.reset_optimizer = False
    config.save_optimizer = True
    config.save_imgs_every = 100
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 2} ,port=29505, start_method="spawn",clear_cache=False, kill_nvidia=False, zulip=True, ping='@**John Dean**')
#@launch(ddp=0,start_method='spawn')
def Dec4_resconvgibbs():
    config.prefix = '_diffusion3'
    config.nope = 0
    #config.gpus = '3'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    import evals.package_neo as pack
    forecaster = pack.get_joansucks()
    mesh = forecaster.config.outputs[0]

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [24]
    data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                            timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False


    # turn off gradients for forecaster
    for p in forecaster.parameters():
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
        conditioning_input_channels=1024,
        conditioning_channels=512
    )

    print(diffuser)

    model = ForecastCombinedDiffusion(forecaster=forecaster,diffuser=diffuser, schedule='linear',append_input=True)

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
    config.lr_sched.cosine_bottom = None
    config.lr_sched.div_factor = 1
    config.save_imgs_every = 100000000
    config.strict_load = False
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'glass': 6}, port=29501, start_method="spawn",clear_cache=False, kill_nvidia=False)#, zulip=True, ping='@**Jack Michaels**')#,validate=False)
#@launch(ddp=0,start_method='spawn')
def Dec13_stackedconvplus():
    #config.prefix = '_penguins'
    # config.nope = 0
    #config.gpus = '0'
    config.resume = "_"+config.activity.replace("_","-")+"_"
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
    #timesteps = [6,24]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            #timesteps=timesteps,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = True

    import model_latlon.top as top
    import model_latlon.encoder as encoders
    import model_latlon.decoder as decoders
    conf = top.ForecastModelConfig(
        data.config.inputs,
        latent_size = 1280,
        affine = True,
        pr_depth = [8],
        encdec_tr_depth = 2,
        tr_embedding = 'rotary',
        patch_size = (4,8,8),
        weight_eps=0.02,
    )
    
    encoder = encoders.SimpleConvEncoder(imesh,conf)
    decoder = decoders.StackedConvPlusDecoder(omesh,conf)
    model = top.ForecastModel(imesh,conf,encoders=[encoder],decoders=[decoder])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.save_imgs_every = 250
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    #config.shampoo.dim = 1024
    config.reset_optimizer = False
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 50_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.3e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.lr_sched.make_plots()
    config.save_imgs_every = 100
    
    config.strict_load = False
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(nodes={'barceloneta': 5, 'baga.fast': 5} ,port=29501, start_method="spawn",clear_cache=False, kill_nvidia=False)#,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
@launch(ddp=0,start_method='spawn')
def johnsucks():
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

    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvEncoder, SimpleConvDecoder, ResConvEncoder, ResConvDecoder
    conf = ForecastModelConfig(
        imesh,
        latent_size = 1280,
        pr_dims = [48, 192, 512],
        affine = True,
        pr_depth = 10,
        encdec_tr_depth = 4,
        oldpr = True,
        tr_embedding = 'rotary',
    )
    encoder = ResConvEncoder(imesh,conf)
    decoder = ResConvDecoder(omesh,conf)
    model = ForecastModel(imesh,conf,encoders=[encoder],decoders=[decoder])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.disregard_buffer_checksum = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    #config.shampoo.dim = 1024
    config.reset_optimizer = False
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 50_000
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
def Dec5_gibbs_multitimestep():
    config.prefix = '_diffusion3'
    config.nope = 0
    config.gpus = '0'
    config.resume = "_"+config.activity.replace("_","-")+"_"

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
    config.lr_sched.cosine_period = 100_000
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.lr_sched.restart_warmup_end_step = 0
    config.lr_sched.div_factor = 1
    config.lr_sched.cosine_bottom = None
    config.save_imgs_every = 100000000
    config.strict_load = False
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'muir': 3} ,port=29503, start_method="spawn",clear_cache=False, kill_nvidia=False, zulip=True, ping='@**John Dean**')
#@launch(ddp=0,start_method='spawn')
def Dec4_gibbs():
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
    config.lr_sched.cosine_period = 100_000
    config.lr_sched.warmup_end_step = 100
    config.lr_sched.lr = 1e-4
    config.lr_sched.restart_warmup_end_step = 0
    config.lr_sched.cosine_bottom = None
    config.lr_sched.div_factor = 1
    config.save_imgs_every = 100000000
    config.strict_load = False
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



if __name__ == '__main__':
    run(locals().values())
