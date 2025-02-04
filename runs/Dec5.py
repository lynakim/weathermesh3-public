import sys
sys.path.append('.')
from runs.launch import *

if am_i_torchrun():
    from train.trainer import *

from train.trainer import * # <--- if using DDP comment this out


@launch(nodes={'muir': 5} ,port=29513, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')
#@launch(ddp=False)
def Feb3_operational_stackedconvplus_hres_gfs():
    #config.nope = True
    
    tdates = get_dates([(D(2021, 3, 23), D(2024, 2, 1))])
    
    from evals.package_neo import get_operational_stackedconvplus_finetime
    stackedconvplus = get_operational_stackedconvplus_finetime()
    
    hres_mesh = stackedconvplus.encoders[0].mesh
    gfs_mesh = stackedconvplus.encoders[1].mesh
    omesh = stackedconvplus.decoders[0].mesh
    timesteps = [0, 6, 24, 72]
    data = data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(hres_mesh),
            AnalysisDataset(gfs_mesh, gfs=True),
        ],
        outputs=[
            AnalysisDataset(omesh),
        ],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    for param in stackedconvplus.parameters():
        param.requires_grad = False

    for param in stackedconvplus.encoders.parameters():
        param.requires_grad = True
    
    config.strict_load = True
    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo' #'adam' 
    config.shampoo.dim = 4096
    config.reset_optimizer = True
    config.reset_steps_on_resume = True#True#False#True
    config.save_optimizer = True

    config.lr_sched.cosine_period = 15_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = (0.3e-3) / 5
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100_000
    #config.profile = not config.nope
    
    w = WeatherTrainer(conf=config,model=stackedconvplus,data=data)
    w.run()
    
@launch(nodes={'stinson': 1} ,port=29514, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')#,validate=False)
#@launch(ddp=False)
def Jan31_debug_test_embedding():
    #config.nope = True
    config.gpus = '2-3'
    
    tdates = get_dates([(D(1979, 1, 23), D(2024, 2, 1))])
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0]

    from meshes import MicrowaveData
    mesh_1bamua = MicrowaveData("1bamua") # atms,1bamua
    mesh_atms = MicrowaveData("atms") # atms,1bamua
    
    from datasets import AnalysisDataset, MicrowaveDataset
    data = WeatherDataset(DataConfig(
        inputs=[
            MicrowaveDataset(mesh_1bamua, is_required=True),
            MicrowaveDataset(mesh_atms, is_required=True), 
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        timesteps=timesteps,
        only_at_z=[0,6,12,18],
        requested_dates = tdates
        ))

    conf = ForecastModelConfig(
        None,
        outputs=[omesh],
        #checkpoint_type="torch",
        encdec_tr_depth=4,
        latent_size=896,
        window_size=(3,5,7),
    )

    from model_latlon.decoder import SimpleConvDecoder
    from model_latlon.da_transformer import DAOnlyModel, DATransformer
    from model_latlon.da_encoders import PointObPrEncoder

    # for when doing transformer
    model = DAOnlyModel(
        conf,
        anl_encoders=[],
        obs_encoders=[PointObPrEncoder(mesh_1bamua, conf),
                      PointObPrEncoder(mesh_atms, conf)],
        da_transformer= DATransformer(conf),
        decoders=[SimpleConvDecoder(omesh, conf)]
    )

    config.find_unused_params = False
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
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    #return w
    w.run()

@launch(nodes={'glass': 2} ,port=29514, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')#,validate=False)
#@launch(ddp=False)
def Jan30_test_windborne():
    #config.nope = True
    config.gpus = '3-4'
    
    tdates = get_dates([(D(1979, 1, 23), D(2024, 2, 1))])
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0]

    from meshes import WindborneData
    windborne_mesh = WindborneData()
    
    from datasets import AnalysisDataset, WindborneDataset
    data = WeatherDataset(DataConfig(
        inputs=[
            WindborneDataset(windborne_mesh, is_required=True), 
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        timesteps=timesteps,
        only_at_z=[0,6,12,18],
        requested_dates = tdates
        ))

    conf = ForecastModelConfig(
        None,
        outputs=[omesh],
        #checkpoint_type="torch",
        encdec_tr_depth=4,
        latent_size=896,
        window_size=(3,5,7),
    )

    from model_latlon.decoder import SimpleConvDecoder
    from model_latlon.da_transformer import DAOnlyModel, DATransformer
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    # for when doing transformer
    model = DAOnlyModel(
        conf,
        anl_encoders=[],
        obs_encoders=[PointObPrEncoder(windborne_mesh, conf)],
        da_transformer= DATransformer(conf),
        decoders=[SimpleConvDecoder(omesh, conf)]
    )

    config.find_unused_params = False
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
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    #return w
    w.run()

@launch(nodes={'muir': 6} ,port=29513, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')#,validate=False)
#@launch(nodes={'muir': 6}, kill_nvidia=False)
def Jan24_test_adpupa():
    #config.nope = 0
    #config.gpus = '2'
    tdates = get_dates([(D(2023, 1, 1), D(2024, 10, 1))])
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0]

    from meshes import MicrowaveData, BalloonData, SurfaceData, SatwindData, RadiosondeData
    mesh_1bamua = MicrowaveData("1bamua") # atms,1bamua
    mesh_atms = MicrowaveData("atms") # atms,1bamua
    from datasets import AnalysisDataset, IgraDataset, MicrowaveDataset, SurfaceDataset, SatwindDataset, RadiosondeDataset
    igra_mesh = BalloonData()
    satwindmesh = SatwindData()
    radiosondemesh = RadiosondeData()
    mesh_sfc = SurfaceData()
    
    data = WeatherDataset(DataConfig(
        inputs=[
            MicrowaveDataset(mesh_1bamua, is_required=True), 
            MicrowaveDataset(mesh_atms, is_required=True), 
            IgraDataset(igra_mesh, is_required=True), 
            SurfaceDataset(mesh_sfc, is_required=True), 
            SatwindDataset(satwindmesh, is_required=True),
            RadiosondeDataset(radiosondemesh, is_required=True),
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
    )

    from model_latlon.decoder import SimpleConvDecoder
    from model_latlon.da_transformer import DAOnlyModel, DATransformer
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder, PointObSfcEncoder

    model = DAOnlyModel(
        conf,
        anl_encoders=[],
        obs_encoders=[MicrowaveEncoder(mesh_1bamua, conf, background='full'), 
                      MicrowaveEncoder(mesh_atms, conf, background='full'), 
                      PointObPrEncoder(igra_mesh, conf, background='full'), 
                      PointObSfcEncoder(mesh_sfc, conf, background=None),
                      PointObPrEncoder(satwindmesh, conf),
                      PointObPrEncoder(radiosondemesh, conf)],
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
    config.shampoo.dim = 1024
    config.reset_optimizer = True
    config.reset_steps_on_resume = False#True#False#True
    config.save_optimizer = False
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
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    #return w
    w.run()

@launch(nodes={'stinson': 1}, start_method='spawn', clear_cache=False, kill_nvidia=False)
#@launch(ddp=0)
def Jan21_test_satwnd():
    config.gpus = '2'
    config.nope = True
    
    from evals.package_neo import get_joanrealtimesucks_hresonly
    joansucks = get_joanrealtimesucks_hresonly()
    imesh = joansucks.encoders[0].mesh
    imesh.hour_offset = 6
    omesh = joansucks.decoders[0].mesh
    satwndmesh = meshes.SatwindData()
    
    from datasets import SatwindDataset
    tdates = get_dates((D(2023, 1, 4), D(2023, 1, 31)))
    timesteps = [0, 24]
    data = data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
            SatwindDataset(satwndmesh, is_required=True),
        ],
        outputs=[
            AnalysisDataset(omesh),
        ],
        timesteps=timesteps,
        only_at_z=[0,6,12,18],
        requested_dates = tdates
    ))
    
    for param in joansucks.encoders.parameters():
        param.requires_grad = False
    for param in joansucks.decoders.parameters():
        param.requires_grad = False
    for param in joansucks.processors.parameters():
        param.requires_grad = False
    
    conf = joansucks.config
    conf.da_depth = 8
    conf.checkpoint_type = "matepoint_sync"
    from model_latlon.da_transformer import DATransformer, DAForecastModel2
    from model_latlon.da_encoders import PointObPrEncoder
    model = DAForecastModel2(
        conf,
        anl_encoders=[joansucks.encoders[0]],
        obs_encoders=[PointObPrEncoder(satwndmesh, conf)],
        da_transformer=DATransformer(conf),
        decoders=[joansucks.decoders[0]],
        processors={'6': joansucks.processors['6']}
        )
    
    config.strict_load = True
    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    #config.compute_Bcrit_every = np.nan
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 1024
    config.find_unused_parameters=True
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

@launch(nodes={'miramar': 6, 'bimini': 6}, start_method='spawn', clear_cache=False, kill_nvidia=True, zulip=True, ping='@**Jack Michaels**')
#@launch(ddp=False)
def Jan16_operational_stackedconvplus_finetune(): 
    
    config.gpus = '0-5'
    #config.nope = True
    config.resume = '/huge/deep/runs/run_Jan16-operational_stackedconvplus_finetune_20250116-103413/model_epoch4_iter593988_step49499_loss0.011.pt'
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h',
             'tc-maxws', 'tc-minp'] # New TC outputs!

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    timesteps = [0,6,12,18,24,30,36,42,48,60,72,78,96,120,144]
    data = data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ],
        outputs=[
            AnalysisDataset(omesh),
        ],
        timesteps=timesteps,
        random_timestep_subset=5,
        requested_dates = tdates
    ))

    import model_latlon.top as top
    import model_latlon.encoder as encoders
    import model_latlon.decoder as decoders
    conf = top.ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        latent_size = 1440, # Highly composite number, larger latent space (was 1280)
        affine = True,
        pr_depth = [8],
        encdec_tr_depth = 3, # More Transformers! (was 2)
        tr_embedding = 'rotary',
        patch_size = (4,8,8),
        use_pole_convs = False,
    )
    
    encoder = encoders.SimpleConvEncoder(imesh, conf)
    decoder = decoders.StackedConvPlusDecoder(omesh, conf)
    model = top.ForecastModel(
        conf,
        encoders=[encoder],
        decoders=[decoder]
    )
    
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
    config.shampoo.dim = 4096
    config.reset_optimizer = True # False
    config.reset_steps_on_resume = True
    config.save_optimizer = True
    config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 16_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = (0.3e-3) / 5
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.lr_sched.make_plots()
    
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(ddp=False, port=29513, kill_nvidia=False)
def Jan9_test_yamaka(): 
    config.gpus = '2'
    #config.nope = True
    config.resume = "/huge/deep/runs/run_Dec28-operational-stackedconvplus_20241227-221550/model_epoch1_iter253188_step21099_loss0.015.pt"
    config.prefix = '_profile'
    config.strict_load = False

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h',
             'tc-maxws', 'tc-minp'] # New TC outputs!

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ],
        outputs=[
            AnalysisDataset(omesh),
        ],
        requested_dates = tdates
    ))
    
    config.lr_sched.schedule_dts = True

    import model_latlon.top as top
    import model_latlon.encoder as encoders
    import model_latlon.decoder as decoders
    conf = top.ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        latent_size = 1440, # Highly composite number, larger latent space (was 1280)
        affine = True,
        pr_depth = [8],
        encdec_tr_depth = 3, # More Transformers! (was 2)
        tr_embedding = 'rotary',
        patch_size = (4,8,8),
        checkpoint_type="torch",
    )

    encoder = encoders.SimpleConvEncoder(imesh, conf)
    decoder = decoders.StackedConvPlusDecoder(omesh, conf)
    model = top.ForecastModel(
        conf,
        encoders=[encoder],
        decoders=[decoder]
    )

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
    config.shampoo.dim = 4096
    config.reset_optimizer = False
    config.save_optimizer = True
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

    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'barceloneta': 1}, ddp=False, zulip=True, ping='@**Jack Michaels**')
def Jan7_heavy_yamaka(): 
    config.gpus = '4'
    #config.nope = True
    config.resume = "/huge/deep/runs/run_Dec28-operational-stackedconvplus_20241227-221550/model_epoch3_iter386388_step32199_loss0.010.pt"
    config.prefix = '_yamaka_test'
    config.strict_load = False
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h',
             'tc-maxws', 'tc-minp'] # New TC outputs!

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            requested_dates = tdates
                                            ))
    
    config.lr_sched.schedule_dts = True

    import model_latlon.top as top
    import model_latlon.encoder as encoders
    import model_latlon.decoder as decoders
    conf = top.ForecastModelConfig(
        data.config.inputs,
        latent_size = 1440, # Highly composite number, larger latent space (was 1280)
        affine = True,
        pr_depth = [8],
        encdec_tr_depth = 3, # More Transformers! (was 2)
        tr_embedding = 'rotary',
        patch_size = (4,8,8),
        weight_eps = 0.02,
        checkpoint_type="torch",
    )
    
    encoder = encoders.SimpleConvEncoder(imesh, conf)
    decoder = decoders.StackedConvPlusDecoder(omesh, conf)
    model = top.ForecastModel(
        conf,
        encoders=[encoder],
        decoders=[decoder]
    )
    
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
    config.shampoo.dim = 4096
    config.reset_optimizer = False
    config.save_optimizer = True
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
    
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(nodes={'barceloneta': 1}, start_method='spawn', clear_cache=False, kill_nvidia=False)#, zulip=True, ping='@**Jack Michaels**')
@launch(ddp=False, zulip=True, ping='@**Jack Michaels**')
def Jan7_mclaren_yamaka(): 
    config.gpus = '5'
    #config.nope = True
    config.resume = "/huge/deep/runs/run_Dec28-operational-stackedconvplus_20241227-221550/model_epoch3_iter386388_step32199_loss0.010.pt"
    config.prefix = '_yamaka_test'
    config.strict_load = False
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h',
             'tc-maxws', 'tc-minp'] # New TC outputs!

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            requested_dates = tdates
                                            ))
    
    config.lr_sched.schedule_dts = True

    import model_latlon.top as top
    import model_latlon.encoder as encoders
    import model_latlon.decoder as decoders
    conf = top.ForecastModelConfig(
        data.config.inputs,
        latent_size = 1440, # Highly composite number, larger latent space (was 1280)
        affine = True,
        pr_depth = [8],
        encdec_tr_depth = 3, # More Transformers! (was 2)
        tr_embedding = 'rotary',
        patch_size = (4,8,8),
        weight_eps = 0.02,
        checkpoint_type="torch",
    )
    
    encoder = encoders.SimpleConvEncoder(imesh, conf)
    decoder = decoders.StackedConvPlusDecoder(omesh, conf)
    model = top.ForecastModel(
        conf,
        encoders=[encoder],
        decoders=[decoder]
    )
    
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
    config.shampoo.dim = 4096
    config.reset_optimizer = False
    config.save_optimizer = True
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
    
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(ddp=False, kill_nvidia=False)
#@launch(nodes={'stinson': 2}, port=25007, start_method="spawn", kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')
def Jan3_test_refactor():
    config.nope = True
    #config.gpus = '0-1'
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_pad=len(extra), 
        input_levels=levels_medium,
        levels=levels_medium)
    
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_medium)
    '''
    output_mesh_2 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_medium)'''
    #output_mesh_2 = meshes.TCRegionalIntensities()
    
    timesteps = [6]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1],#, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    from model_latlon.top import ForecastModel, ForecastModelConfig
    from model_latlon.decoder import SimpleConvDecoder, RegionalTCDecoder
    from model_latlon.encoder import SimpleConvEncoder
    
    conf = ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        #checkpoint_type="torch",
        encdec_tr_depth=4,
        latent_size=896,
        window_size=(3,5,7),
        weight_eps=0.01,
    )
    
    model = ForecastModel(
        conf,
        encoders=[SimpleConvEncoder(input_mesh, conf)],
        decoders=[
            SimpleConvDecoder(output_mesh_1, conf), 
            SimpleConvDecoder(output_mesh_1, conf, decoder_loss_weight=0.25),
            ])
    
    config.diffusion = False
    config.ignore_train_safeguard = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'adam'#'shampoo'
    config.shampoo.dim = 4096 # 256
    config.shampoo.version = 'old'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 22_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.loss_consts_override = {6:1,24:1}
    config.find_unused_params = True

    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(nodes={'baga': 5}, port=29500, start_method="spawn",clear_cache=False, kill_nvidia=True, zulip=True, ping='@**Jack Michaels**')
@launch(nodes={'miramar': 6, 'bimini': 6}, start_method='spawn', clear_cache=False, kill_nvidia=True, zulip=True, ping='@**Jack Michaels**')
#@launch(ddp=False)
def Dec28_operational_stackedconvplus(): # Duplicated config from runs/jd.py Dec13_stackedconvplus
    
    config.gpus = '0-5'
    # config.nope = True
    config.resume = "_"+config.activity.replace("_","-")+"_"
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h',
             'tc-maxws', 'tc-minp'] # New TC outputs!

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = True

    import model_latlon.top as top
    import model_latlon.encoder as encoders
    import model_latlon.decoder as decoders
    conf = top.ForecastModelConfig(
        data.config.inputs,
        latent_size = 1440, # Highly composite number, larger latent space (was 1280)
        affine = True,
        pr_depth = [8],
        encdec_tr_depth = 3, # More Transformers! (was 2)
        tr_embedding = 'rotary',
        patch_size = (4,8,8),
        weight_eps = 0.02,
    )
    
    encoder = encoders.SimpleConvEncoder(imesh, conf)
    decoder = decoders.StackedConvPlusDecoder(omesh, conf)
    model = top.ForecastModel(
        conf,
        encoders=[encoder],
        decoders=[decoder]
    )
    
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
    config.shampoo.dim = 4096
    config.reset_optimizer = False
    config.save_optimizer = True
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
    
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'baga': 5}, port=29500, start_method="spawn",clear_cache=False, kill_nvidia=True, zulip=True, ping='@**Jack Michaels**')
#@launch(ddp=0,start_method='spawn')
def Dec24_operational_stackedconvplus_test(): # Duplicated config from runs/jd.py Dec13_stackedconvplus
    #config.prefix = '_penguins'
    #config.nope = True
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
    #config.lr_sched.schedule_dts_warmup_end_step = 100
    #config.num_workers = 0
    #config.prefetch_factor = None

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
    
    #print(model)

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
    
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'bimini': 2}, port=29501, start_method="spawn", clear_cache=False, kill_nvidia=True, zulip=True, ping='@**Jack Michaels**')
def Dec23_TCregionalio(): 
    # config.nope = True
    config.gpus = '2-3'
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_medium)
    
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_medium)
    output_mesh_2 = meshes.TCRegionalIntensities()
    
    timesteps = [6, 24]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    import model_latlon.top as top
    import model_latlon.encoder as encoders
    import model_latlon.decoder as decoders
    
    conf = top.ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth = 4,
        latent_size=896,
        window_size=(3,5,7),
        weight_eps=0.01,
    )
    
    encoder = encoders.SimpleConvEncoder(input_mesh, conf)
    decoder_standard = decoders.StackedConvPlusDecoder(output_mesh_1, conf)
    decoder_tc_regional = decoders.RegionalTCDecoder(output_mesh_2, conf, region_radius=2, hidden_dim=128, decoder_loss_weight=(0.25 / 4))
    
    model = top.ForecastModel(
        input_mesh, 
        conf,
        encoders=[encoder],
        decoders=[decoder_standard, decoder_tc_regional]
    )

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.save_imgs_every = 100
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.HALF = True
    config.optim = 'adam' # 'shampoo' 
    # config.shampoo.version = 'old'
    config.shampoo.dim = 4096 # 256
    config.lr_sched.cosine_period = 30_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.reset_optimizer = True
    config.save_optimizer = False
    config.latent_l2 = 1e-4
    config.loss_consts_override = {6:1,24:1}
    config.find_unused_params = True
    
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'bimini': 2}, port=29502, start_method="spawn", clear_cache=False, kill_nvidia=True, zulip=True, ping='@**Jack Michaels**')
def Dec23_TCregional72(): 
    # config.nope = True
    config.gpus = '0-1'
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_pad=len(extra), 
        input_levels=levels_medium,
        levels=levels_medium)
    
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_medium)
    output_mesh_2 = meshes.TCRegionalIntensities()
    
    timesteps = [6, 24, 72]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    import model_latlon.top as top
    import model_latlon.encoder as encoders
    import model_latlon.decoder as decoders
    
    conf = top.ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth = 4,
        latent_size=896,
        window_size=(3,5,7),
        weight_eps=0.01,
    )
    
    encoder = encoders.SimpleConvEncoder(input_mesh, conf)
    decoder_standard = decoders.StackedConvPlusDecoder(output_mesh_1, conf)
    decoder_tc_regional = decoders.RegionalTCDecoder(output_mesh_2, conf, region_radius=2, hidden_dim=128, decoder_loss_weight=(0.25 / 4))
    
    model = top.ForecastModel(
        input_mesh, 
        conf,
        encoders=[encoder],
        decoders=[decoder_standard, decoder_tc_regional]
    )

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.save_imgs_every = 100
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.HALF = True
    config.optim = 'adam' # 'shampoo' 
    # config.shampoo.version = 'old'
    config.shampoo.dim = 4096 # 256
    config.lr_sched.cosine_period = 15_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 2e-4
    config.reset_optimizer = True
    config.save_optimizer = False
    config.latent_l2 = 1e-4
    config.loss_consts_override = {6:1,24:1}
    config.find_unused_params = True
    
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'stinson': 1}, port=29505, start_method="spawn", kill_nvidia=False)#, zulip=True, ping='@**Jack Michaels**')
def Dec11_test_TC_code_changes():
    config.nope = True
    config.gpus = '3-4'
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_pad=len(extra), 
        input_levels=levels_medium,
        levels=levels_medium)
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_medium)
    output_mesh_2 = meshes.TCRegionalLocations()
    
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvEncoder
    from model_latlon.decoder import SimpleConvPlusDecoder, RegionalTCDecoder # StackedConvPlusDecoder
    
    conf = ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth=4,
        latent_size=896,
        window_size=(3,5,7)
    )
    
    model = ForecastModel(
        input_mesh, 
        conf,
        encoders=[SimpleConvEncoder(input_mesh, conf)],
        decoders=[
            SimpleConvPlusDecoder(input_mesh, conf),
            RegionalTCDecoder(input_mesh, 
                              conf, 
                              region_radius=2, 
                              tr_depth=2, 
                              tr_window_size=(3, 5, 7))
            ])
    
    config.diffusion = False
    config.ignore_train_safeguard = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'adam'#'shampoo'
    config.shampoo.dim = 4096 # 256
    config.shampoo.version = 'old'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 25_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    config.decoder_loss_weights = [1, 0.25]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'miramar': 1}, port=29505, start_method="spawn", kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')
def Dec11_TCregionalarch_test():
    #config.nope = True
    config.gpus = '2-3'
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_pad=len(extra), 
        input_levels=levels_medium,
        levels=levels_joank)
    
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_joank)
    output_mesh_2 = meshes.TCRegionalLocations()
    
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvDecoder, RegionalTCDecoder
    conf = ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth=4,
        oldenc=True,
        latent_size=896,
        window_size=(3,5,7),
        decoder_types=[SimpleConvDecoder, RegionalTCDecoder],
        decoder_configs=[
            SimpleNamespace(),
            SimpleNamespace(
                bounding_box = (2, 2), # (90, 180) latent space dims, so 2x2 is actually 16x16 in 740x1440 space
            )
        ]
    )
    
    model = ForecastModel(input_mesh, conf)
    
    config.diffusion = False
    config.ignore_train_safeguard = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'adam'#'shampoo'
    config.shampoo.dim = 4096 # 256
    config.shampoo.version = 'old'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 22_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1,24:1}
    config.decoder_loss_weights = [1, 0.25]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'miramar': 1}, port=29504, start_method="spawn", kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')
def Dec10_TCregional128_test():
    # config.nope = True
    config.gpus = '0-1'
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_pad=len(extra), 
        input_levels=levels_medium,
        levels=levels_joank)
    
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_joank)
    output_mesh_2 = meshes.TCRegionalLocations()
    
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvDecoder, RegionalTCDecoder
    conf = ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth=4,
        oldenc=True,
        latent_size=896,
        window_size=(3,5,7),
        decoder_types=[SimpleConvDecoder, RegionalTCDecoder],
        decoder_configs=[
            SimpleNamespace(),
            SimpleNamespace(
                bounding_box = (2, 2), # (90, 180) latent space dims, so 2x2 is actually 16x16 in 740x1440 space
                hidden_dim = 128,
            )
        ]
    )
    
    model = ForecastModel(input_mesh, conf)
    
    config.diffusion = False
    config.ignore_train_safeguard = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'adam'#'shampoo'
    config.shampoo.dim = 4096 # 256
    config.shampoo.version = 'old'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 22_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1,24:1}
    config.decoder_loss_weights = [1, 0.25]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'miramar': 1}, port=29503, start_method="spawn", kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')
def Dec10_TCregionalio_test():
    # config.nope = True
    config.gpus = '1-2'
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_joank)
    
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_joank)
    output_mesh_2 = meshes.TCRegionalLocations()
    
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvDecoder, RegionalTCDecoder
    conf = ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth=4,
        oldenc=True,
        latent_size=896,
        window_size=(3,5,7),
        decoder_types=[SimpleConvDecoder, RegionalTCDecoder],
        decoder_configs=[
            SimpleNamespace(),
            SimpleNamespace(
                bounding_box = (2, 2), # (90, 180) latent space dims, so 2x2 is actually 16x16 in 740x1440 space
                hidden_dim = 256,
            )
        ]
    )
    
    model = ForecastModel(input_mesh, conf)
    
    config.diffusion = False
    config.ignore_train_safeguard = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'adam'#'shampoo'
    config.shampoo.dim = 4096 # 256
    config.shampoo.version = 'old'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 22_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1,24:1}
    config.decoder_loss_weights = [1, 0.25]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'miramar': 1}, port=29502, start_method="spawn", kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')
def Dec10_TCregional72_test():
    #config.nope = True
    config.gpus = '3-4'
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_pad=len(extra), 
        input_levels=levels_medium,
        levels=levels_joank)
    
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_joank)
    output_mesh_2 = meshes.TCRegionalLocations()
    
    timesteps = [6,24,72]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvDecoder, RegionalTCDecoder
    conf = ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth=4,
        oldenc=True,
        latent_size=896,
        window_size=(3,5,7),
        decoder_types=[SimpleConvDecoder, RegionalTCDecoder],
        decoder_configs=[
            SimpleNamespace(),
            SimpleNamespace(
                bounding_box = (2, 2), # (90, 180) latent space dims, so 2x2 is actually 16x16 in 740x1440 space
                hidden_dim = 256,
            )
        ]
    )
    
    model = ForecastModel(input_mesh, conf)
    
    config.diffusion = False
    config.ignore_train_safeguard = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'adam'#'shampoo'
    config.shampoo.dim = 4096 # 256
    config.shampoo.version = 'old'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 22_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1,24:1}
    config.decoder_loss_weights = [1, 0.25]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'miramar': 1}, port=29501, start_method="spawn", kill_nvidia=False)#, zulip=True, ping='@**Jack Michaels**')
def Dec10_TCregionallarger_test():
    #config.nope = True
    config.gpus = '4-5'
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_pad=len(extra), 
        input_levels=levels_medium,
        levels=levels_joank)
    
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_joank)
    output_mesh_2 = meshes.TCRegionalLocations()
    
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvDecoder, RegionalTCDecoder
    conf = ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth=4,
        oldenc=True,
        latent_size=896,
        window_size=(3,5,7),
        decoder_types=[SimpleConvDecoder, RegionalTCDecoder],
        decoder_configs=[
            SimpleNamespace(),
            SimpleNamespace(
                bounding_box = (4, 4), # (90, 180) latent space dims, so 2x2 is actually 16x16 in 740x1440 space
                hidden_dim = 256,
            )
        ]
    )
    
    model = ForecastModel(input_mesh, conf)
    
    config.diffusion = False
    config.ignore_train_safeguard = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'adam'#'shampoo'
    config.shampoo.dim = 4096 # 256
    config.shampoo.version = 'old'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched.cosine_period = 22_000
    config.lr_sched.warmup_end_step = 1000
    config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1,24:1}
    config.decoder_loss_weights = [1, 0.25]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'miramar': 1}, start_method="spawn", kill_nvidia=False, zulip=True, ping='@**Jack Michaels**')
def Dec10_TCregional_test():
    #config.nope = True
    config.gpus = '5-6'
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_pad=len(extra), 
        input_levels=levels_medium,
        levels=levels_joank)
    
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_joank)
    output_mesh_2 = meshes.TCRegionalLocations()
    
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvDecoder, RegionalTCDecoder
    conf = ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth=4,
        oldenc=True,
        latent_size=896,
        window_size=(3,5,7),
        decoder_types=[SimpleConvDecoder, RegionalTCDecoder],
        decoder_configs=[
            SimpleNamespace(),
            SimpleNamespace(
                bounding_box = (2, 2), # (90, 180) latent space dims, so 2x2 is actually 16x16 in 740x1440 space
                hidden_dim = 256,
            )
        ]
    )
    
    model = ForecastModel(input_mesh, conf)
    
    config.diffusion = False
    config.ignore_train_safeguard = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'adam'
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
    config.decoder_loss_weights = [1, 0.25]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()
    

@launch(ddp=False, kill_nvidia=False)
def Dec5_TCregional_test(): # Dec6_nether_decoding()
    config.nope = True
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_pad=len(extra), 
        input_levels=levels_medium,
        levels=levels_joank)
    
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_joank)
    output_mesh_2 = meshes.TCRegionalLocations()
    
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvDecoder, RegionalTCDecoder
    conf = ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth=4,
        oldenc=True,
        latent_size=896,
        window_size=(3,5,7),
        decoder_types=[SimpleConvDecoder, RegionalTCDecoder],
        decoder_configs=[
            SimpleNamespace(),
            SimpleNamespace(
                bounding_box = (2, 2), # (90, 180) latent space dims, so 2x2 is actually 16x16 in 740x1440 space
                hidden_dim = 256,
            )
        ]
    )
    
    model = ForecastModel(input_mesh, conf)
    
    config.diffusion = False
    config.ignore_train_safeguard = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
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
    config.decoder_loss_weights = [1, 0.25]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()
    

if __name__ == '__main__':
    run(locals().values())
