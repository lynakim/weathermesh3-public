import sys
sys.path.append('.')
from runs.launch import *

if am_i_torchrun():
    from train.trainer import *

from train.trainer import * # <--- comment this back in if not using DDP
from torch.utils import checkpoint as torch_checkpoint


@launch(nodes={'miramar': 6} ,port=29505, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Feb2_bigens_nomean():
    #config.prefix = '_penguins'
    config.nope = 0
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    timesteps = [0,6,24,48]#,36]
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        only_at_z=[0,3,6,9,12,15,18,21],
        requested_dates = tdates
        ))

    config.lr_sched.schedule_dts = True

    from model_latlon.top_ens import EnsembleForecastModel
    from model_latlon.top import ForecastModelConfig
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 768,
        pr_depth = 8,
        encdec_tr_depth = 2,
        patch_size=(4, 8, 8),
        n_chunks_dec=20,
        dec_sublatent=96,
        deconv_mlp_dim=384,
        deeper_m2g=True,
        ens_nomean=True
    )
    from model_latlon.heal import HealLatentMesh, HealMesh, SimpleHealEncoder, GNNyHealDecoder, IcoSlide3D
    hmesh = HealMesh(8, do_output=False)
    hlmesh = HealLatentMesh(depth=5, D=5, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = GNNyHealDecoder(omesh, conf, compile=False)
    model = EnsembleForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    #config.slow_start = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.latent_l2 = 1e-4
    config.shampoo.dim = 4096
    config.reset_optimizer = True# <-------------------------
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
    config.lr_sched.num_random_subset_max = 4
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    config.num_workers = 2
    config.slow_start = True
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'miramar': 1} ,port=29505, start_method="spawn",clear_cache=False)#,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Feb2_enscp():
    #config.prefix = '_penguins'
    config.nope = 1
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    timesteps = [0,6,24,48]#,36]
    timesteps = [0,6,12]
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        only_at_z=[0,3,6,9,12,15,18,21],
        requested_dates = tdates,
        timesteps=timesteps
        ))

    config.lr_sched.schedule_dts = False#True

    from model_latlon.top_ens import EnsembleForecastModel
    from model_latlon.top import ForecastModelConfig
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 512,
        pr_depth = 2,
        encdec_tr_depth = 2,
        checkpoint_type="torch",
        patch_size=(4, 8, 8),
        n_chunks_dec=20,
        dec_sublatent=96,
        deconv_mlp_dim=384,
        deeper_m2g=True,
        ens_nomean=True
    )
    from model_latlon.heal import HealLatentMesh, HealMesh, SimpleHealEncoder, GNNyHealDecoder, IcoSlide3D
    hmesh = HealMesh(8, do_output=False)
    hlmesh = HealLatentMesh(depth=5, D=5, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = GNNyHealDecoder(omesh, conf, compile=False)
    model = EnsembleForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    #config.slow_start = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    config.optim = 'adam'
    config.latent_l2 = 1e-4
    config.shampoo.dim = 4096
    config.reset_optimizer = True# <-------------------------
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
    config.lr_sched.num_random_subset_max = 4
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    config.num_workers = 2
    config.slow_start = True
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'glass': 5} ,port=29505, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan30_graphbs_bigensyolo():
    #config.prefix = '_penguins'
    config.nope = 0
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    timesteps = [0,6,24,48]#,36]
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        only_at_z=[0,3,6,9,12,15,18,21],
        requested_dates = tdates
        ))

    config.lr_sched.schedule_dts = True

    from model_latlon.top_ens import EnsembleForecastModel
    from model_latlon.top import ForecastModelConfig
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 768,
        pr_depth = 8,
        encdec_tr_depth = 2,
        patch_size=(4, 8, 8),
        n_chunks_dec=20,
        dec_sublatent=96,
        deconv_mlp_dim=384,
        deeper_m2g=True
    )
    from model_latlon.heal import HealLatentMesh, HealMesh, SimpleHealEncoder, GNNyHealDecoder, IcoSlide3D
    hmesh = HealMesh(8, do_output=False)
    hlmesh = HealLatentMesh(depth=5, D=5, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = GNNyHealDecoder(omesh, conf, compile=False)
    model = EnsembleForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    #config.slow_start = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.latent_l2 = 1e-4
    config.shampoo.dim = 4096
    config.reset_optimizer = True# <-------------------------
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
    config.lr_sched.num_random_subset_max = 4
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    config.num_workers = 2
    config.slow_start = True
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'barceloneta': 6} ,port=29505, start_method="spawn")#,clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan28_graphbs_bucketsofrain():
    #config.prefix = '_penguins'
    config.nope = False
    #config.nope = True
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']

    buckets_of_rain = np.append(np.logspace(-2-1/3,1,11)/1000., np.inf) # in meters

    def make_buckets(v, b):
        return [v+"_bucket"+str(i) for i in range(len(b))]

    extra_output += make_buckets("142_lsp", buckets_of_rain)
    extra_output += make_buckets("143_cp", buckets_of_rain)

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank, precip_buckets=buckets_of_rain)
    timesteps = [0,6,24,48]#,36]
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        only_at_z=[0,3,6,9,12,15,18,21],
        requested_dates = tdates
        ))

    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 768,
        pr_depth = 8,
        encdec_tr_depth = 2,
        patch_size=(5, 8, 8),
        dec_sublatent=96,
        deconv_mlp_dim=384,
        deeper_m2g=True
    )
    from model_latlon.heal import HealLatentMesh, HealMesh, SimpleHealEncoder, GNNyHealDecoder, IcoSlide3D
    hmesh = HealMesh(8, do_output=False)
    hlmesh = HealLatentMesh(depth=5, D=6, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = GNNyHealDecoder(omesh, conf)
    model = ForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.slow_start = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.latent_l2 = 1e-4
    #config.latent_l2 = 0
    config.shampoo.dim = 4096
    config.reset_optimizer = False # <---------------------------
    config.save_optimizer = True
    config.reset_steps_on_resume = False
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
    config.num_workers = 3
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'muir': 5} ,port=29505, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan30_graphbs_ensyolo():
    #config.prefix = '_penguins'
    config.nope = 0
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    timesteps = [0,6,24,48]#,36]
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        only_at_z=[0,3,6,9,12,15,18,21],
        requested_dates = tdates
        ))

    config.lr_sched.schedule_dts = True

    from model_latlon.top_ens import EnsembleForecastModel
    from model_latlon.top import ForecastModelConfig
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 512,
        pr_depth = 8,
        encdec_tr_depth = 2,
        patch_size=(4, 8, 8),
        n_chunks_dec=20,
        dec_sublatent=64,
        deconv_mlp_dim=256,
        deeper_m2g=True
    )
    from model_latlon.heal import HealLatentMesh, HealMesh, SimpleHealEncoder, GNNyHealDecoder, IcoSlide3D
    hmesh = HealMesh(8, do_output=False)
    hlmesh = HealLatentMesh(depth=5, D=5, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = GNNyHealDecoder(omesh, conf, compile=False)
    model = EnsembleForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    #config.slow_start = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.latent_l2 = 1e-4
    config.shampoo.dim = 4096
    config.reset_optimizer = True# <-------------------------
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
    config.num_workers = 2
    config.slow_start = True
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'baga': 5} ,port=29505, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan28_graphbs_deepmlp():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)
    timesteps = [0,6,24,48]#,36]
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        only_at_z=[0,3,6,9,12,15,18,21],
        requested_dates = tdates
        ))

    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 768,
        pr_depth = 8,
        encdec_tr_depth = 2,
        patch_size=(5, 8, 8),
        dec_sublatent=64,
        deconv_mlp_dim=256,
        deeper_m2g=True
    )
    from model_latlon.heal import HealLatentMesh, HealMesh, SimpleHealEncoder, GNNyHealDecoder, IcoSlide3D
    hmesh = HealMesh(8, do_output=False)
    hlmesh = HealLatentMesh(depth=5, D=6, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = GNNyHealDecoder(omesh, conf, compile=False)
    model = ForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    #config.slow_start = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.latent_l2 = 1e-4
    config.shampoo.dim = 2048
    config.reset_optimizer = True# <-------------------------
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
    config.num_workers = 2
    config.slow_start = True
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'barceloneta': 6} ,port=29505, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan28_graphbs_gnnswap():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)
    timesteps = [0,6,24]#,36]
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        only_at_z=[0,3,6,9,12,15,18,21],
        timesteps=timesteps,
        requested_dates = tdates
        ))

    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 1024,
        pr_depth = 10,
        encdec_tr_depth = 4,
        patch_size=(5, 8, 8),
    )
    from model_latlon.heal import HealLatentMesh, HealMesh, SimpleHealEncoder, GNNyHealDecoder, IcoSlide3D
    hmesh = HealMesh(8)
    hlmesh = HealLatentMesh(depth=5, D=6, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)

    for param in enc.parameters():
        param.requires_grad = False
    for param in proc.parameters():
        param.requires_grad = False

    dec = GNNyHealDecoder(omesh, conf)
    model = ForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.slow_start = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.latent_l2 = 1e-4
    #config.latent_l2 = 0
    config.shampoo.dim = 2048
    config.reset_optimizer = False
    config.save_optimizer = True
    config.reset_steps_on_resume = False
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    #config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 25_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.125e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 4
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    config.num_workers = 4
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'muir': 5} ,port=29505, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan28_graphbs_gnnyolo():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)
    timesteps = [0,6,24,48]#,36]
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        only_at_z=[0,3,6,9,12,15,18,21],
        requested_dates = tdates
        ))

    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 768,
        pr_depth = 8,
        encdec_tr_depth = 2,
        patch_size=(5, 8, 8),
        dec_sublatent=64,
        deconv_mlp_dim=256
    )
    from model_latlon.heal import HealLatentMesh, HealMesh, SimpleHealEncoder, GNNyHealDecoder, IcoSlide3D
    hmesh = HealMesh(8, do_output=False)
    hlmesh = HealLatentMesh(depth=5, D=6, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = GNNyHealDecoder(omesh, conf, compile=False)
    model = ForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    #config.slow_start = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.latent_l2 = 1e-4
    config.shampoo.dim = 2048
    config.reset_optimizer = False # <-------------------------
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
    config.num_workers = 2
    config.slow_start = True
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'muir': 6} ,port=29505, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan18_graphbs_tinyyolo():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)
    timesteps = [0,6,24,48]#,36]
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        only_at_z=[0,3,6,9,12,15,18,21],
        requested_dates = tdates
        ))

    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 768,
        pr_depth = 8,
        encdec_tr_depth = 2,
        patch_size=(5, 8, 8),
    )
    from model_latlon.heal import HealLatentMesh, HealMesh, SimpleHealEncoder, SimpleHealDecoder, IcoSlide3D
    hmesh = HealMesh(8)
    hlmesh = HealLatentMesh(depth=5, D=6, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = SimpleHealDecoder(omesh, conf, compile=True)
    model = ForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    #config.slow_start = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.latent_l2 = 1e-4
    config.shampoo.dim = 2048
    config.reset_optimizer = True # <-------------------------
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
    config.num_workers = 4
    config.slow_start = True
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'barceloneta': 6} ,port=29505, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan18_graphbs_yolo():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)
    timesteps = [0,6,24,48]#,36]
    
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        only_at_z=[0,3,6,9,12,15,18,21],
        requested_dates = tdates
        ))

    config.lr_sched.schedule_dts = True

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 1024,
        pr_depth = 10,
        encdec_tr_depth = 4,
        patch_size=(5, 8, 8),
    )
    from model_latlon.heal import HealLatentMesh, HealMesh, SimpleHealEncoder, SimpleHealDecoder, IcoSlide3D
    hmesh = HealMesh(8)
    hlmesh = HealLatentMesh(depth=5, D=6, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = SimpleHealDecoder(omesh, conf, compile=True)
    model = ForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])
    print(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.slow_start = True
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.latent_l2 = 1e-4
    #config.latent_l2 = 0
    config.shampoo.dim = 2048
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
    config.lr_sched.num_random_subset_max = 4
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100
    config.weight_eps = 0.02
    config.num_workers = 4
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'glass': 6} ,port=29506, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
#@launch(ddp=0)
def Jan10_daforecast_bigda():
    config.nope = 0
    tdates = get_dates([(D(1979, 1, 23), D(2024, 2, 1))])

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
    from model_latlon.decoder import SimpleConvDecoder
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.da_transformer import DAOnlyModel, DATransformer, DAForecastModel2
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    for param in joansucks.parameters():
        param.requires_grad = False

    conf = joansucks.config
    conf.da_depth = 8
    conf.checkpoint_type = "matepoint_sync"
    model = DAForecastModel2(
        conf,
        anl_encoders=[joansucks.encoders[0]],
        obs_encoders=[MicrowaveEncoder(mesh_1bamua, conf, background='single'), MicrowaveEncoder(mesh_atms, conf, background='single'), PointObPrEncoder(igra_mesh, conf, background='single')],
        da_transformer=DATransformer(conf),
        decoders=[joansucks.decoders[0]],
        processors=[joansucks.processors['6']]
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


@launch(nodes={'glass': 6} ,port=29503, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
#@launch(ddp=0)
def Jan11_daforecast_perturber():
    config.nope = 0
    tdates = get_dates([(D(1979, 1, 23), D(2024, 2, 1))])

    from evals.package_neo import get_joanrealtimesucks_hresonly
    joansucks = get_joanrealtimesucks_hresonly()
    imesh = joansucks.encoders[0].mesh
    imesh.hour_offset = 6
    omesh = joansucks.decoders[0].mesh

    timesteps = [0,24]
    #timesteps = [0]

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
    from model_latlon.decoder import SimpleConvDecoder
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.da_transformer import DAOnlyModel, DATransformer, DAForecastModel2
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    #for param in joansucks.parameters():
    #    param.requires_grad = False
    for param in joansucks.encoders.parameters():
        param.requires_grad = False
    for param in joansucks.decoders.parameters():
        param.requires_grad = False
    for param in joansucks.processors.parameters():
        param.requires_grad = False

    conf = joansucks.config
    conf.da_perturber = 0.1
    conf.checkpoint_type = "matepoint_sync"
    model = DAForecastModel2(
        conf,
        anl_encoders=[joansucks.encoders[0]],
        obs_encoders=[MicrowaveEncoder(mesh_1bamua, conf, background='single'), MicrowaveEncoder(mesh_atms, conf, background='single'), PointObPrEncoder(igra_mesh, conf, background='single')],
        da_transformer=DATransformer(conf),
        decoders=[joansucks.decoders[0]],
        processors=[joansucks.processors['6']]
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


@launch(nodes={'barceloneta': 6} ,port=29503, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
#@launch(ddp=0)
def Jan11_daforecast_test():
    config.nope = 0
    tdates = get_dates([(D(1979, 1, 23), D(2024, 2, 1))])

    from evals.package_neo import get_joanrealtimesucks_hresonly
    joansucks = get_joanrealtimesucks_hresonly()
    imesh = joansucks.encoders[0].mesh
    imesh.hour_offset = 6
    omesh = joansucks.decoders[0].mesh

    timesteps = [0,24]
    #timesteps = [0]

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
    from model_latlon.decoder import SimpleConvDecoder
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.da_transformer import DAOnlyModel, DATransformer, DAForecastModel2
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    #for param in joansucks.parameters():
    #    param.requires_grad = False
    for param in joansucks.encoders.parameters():
        param.requires_grad = False
    for param in joansucks.decoders.parameters():
        param.requires_grad = False
    for param in joansucks.processors.parameters():
        param.requires_grad = False

    conf = joansucks.config
    conf.checkpoint_type = "matepoint_sync"
    model = DAForecastModel2(
        conf,
        anl_encoders=[joansucks.encoders[0]],
        obs_encoders=[MicrowaveEncoder(mesh_1bamua, conf, background='single'), MicrowaveEncoder(mesh_atms, conf, background='single'), PointObPrEncoder(igra_mesh, conf, background='single')],
        da_transformer=DATransformer(conf),
        decoders=[joansucks.decoders[0]],
        processors=[joansucks.processors['6']]
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


@launch(nodes={'stinson': 4} ,port=29506, start_method="spawn",clear_cache=False,kill_nvidia=False)#, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan7_moreobs_satonly_moreobs():
    config.nope = 0
    tdates = get_dates([(D(1979, 1, 23), D(2025, 1, 1))])
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0]

    from model_latlon.da_encoders import MicrowaveData, BalloonData
    mesh_1bamua = MicrowaveData("1bamua", da_timestep_offset=0) # atms,1bamua
    mesh_atms = MicrowaveData("atms", da_timestep_offset=0) # atms,1bamua
    from datasets import AnalysisDataset, IgraDataset, MicrowaveDataset
    #igra_mesh = BalloonData(da_timestep_offset=0)
    
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
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    model = DAOnlyModel(
        conf,
        anl_encoders=[],
        obs_encoders=[MicrowaveEncoder(mesh_1bamua, conf), MicrowaveEncoder(mesh_atms, conf)],
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
    config.shampoo.dim = 2048
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



@launch(nodes={'glass': 6} ,port=29506, start_method="spawn",clear_cache=False,kill_nvidia=False)#, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan7_moreobs_satonly_smalltransformer():
    config.nope = 0
    tdates = get_dates([(D(1979, 1, 23), D(2025, 1, 1))])
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0]

    from model_latlon.da_encoders import MicrowaveData, BalloonData
    mesh_1bamua = MicrowaveData("1bamua", da_timestep_offset=0) # atms,1bamua
    mesh_atms = MicrowaveData("atms", da_timestep_offset=0) # atms,1bamua
    from datasets import AnalysisDataset, IgraDataset, MicrowaveDataset
    #igra_mesh = BalloonData(da_timestep_offset=0)
    
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
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    model = DAOnlyModel(
        conf,
        anl_encoders=[],
        obs_encoders=[MicrowaveEncoder(mesh_1bamua, conf), MicrowaveEncoder(mesh_atms, conf)],
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
    config.shampoo.dim = 2048
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



@launch(nodes={'barceloneta': 3} ,port=29506, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan7_amsu_1h():
    config.nope = 0
    tdates = get_dates([(D(1979, 1, 23), D(2025, 1, 1))])
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0]

    from model_latlon.da_encoders import MicrowaveData, BalloonData
    mesh_1bamua = MicrowaveData("1bamua", da_timestep_offset=0, da_window_size=1) # atms,1bamua
    from datasets import AnalysisDataset, IgraDataset, MicrowaveDataset
    #igra_mesh = BalloonData(da_timestep_offset=0)
    
    data = WeatherDataset(DataConfig(
        inputs=[
            MicrowaveDataset(mesh_1bamua, is_required=True), 

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
    from model_latlon.da_transformer import DAOnlyModel, DATransformer
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    model = DAOnlyModel(
        conf,
        anl_encoders=[],
        obs_encoders=[MicrowaveEncoder(mesh_1bamua, conf)],
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



@launch(nodes={'glass': 6} ,port=29506, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan7_moreobs_satonly():
    config.nope = 0
    tdates = get_dates([(D(1979, 1, 23), D(2025, 1, 1))])
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0]

    from model_latlon.da_encoders import MicrowaveData, BalloonData
    mesh_1bamua = MicrowaveData("1bamua", da_timestep_offset=0) # atms,1bamua
    mesh_atms = MicrowaveData("atms", da_timestep_offset=0) # atms,1bamua
    from datasets import AnalysisDataset, IgraDataset, MicrowaveDataset
    #igra_mesh = BalloonData(da_timestep_offset=0)
    
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
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    model = DAOnlyModel(
        conf,
        anl_encoders=[],
        obs_encoders=[MicrowaveEncoder(mesh_1bamua, conf), MicrowaveEncoder(mesh_atms, conf)],
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
    config.shampoo.dim = 2048
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



@launch(nodes={'barceloneta': 6} ,port=29506, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan10_moreobs_neo_freeze():
    config.nope = 0
    tdates = get_dates([(D(1979, 1, 23), D(2024, 2, 1))])
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0]

    from model_latlon.da_encoders import MicrowaveData, BalloonData
    mesh_1bamua = MicrowaveData("1bamua") # atms,1bamua
    mesh_atms = MicrowaveData("atms") # atms,1bamua
    from datasets import AnalysisDataset, IgraDataset, MicrowaveDataset
    igra_mesh = BalloonData()
    
    data = WeatherDataset(DataConfig(
        inputs=[
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
    )

    from model_latlon.decoder import SimpleConvDecoder
    from model_latlon.da_transformer import DAOnlyModel, DATransformer
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    model = DAOnlyModel(
        conf,
        anl_encoders=[],
        obs_encoders=[MicrowaveEncoder(mesh_1bamua, conf, background='full'), MicrowaveEncoder(mesh_atms, conf, background='full'), PointObPrEncoder(igra_mesh, conf, background='full')],
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



#@launch(ddp=0,start_method='spawn')
@launch(nodes={'stinson': 4} ,port=29501, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Jan6_microwaveonly_sincos():
    config.nope = 0
    tdates = get_dates([(D(1979, 1, 23), D(2025, 1, 1))])
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0]

    from model_latlon.da_encoders import MicrowaveData, BalloonData
    mesh_1bamua = MicrowaveData("1bamua") # atms,1bamua
    from datasets import AnalysisDataset, IgraDataset, MicrowaveDataset
    #igra_mesh = BalloonData(da_timestep_offset=0)
    
    data = WeatherDataset(DataConfig(
        inputs=[
            MicrowaveDataset(mesh_1bamua, is_required=True), 
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
    from model_latlon.da_transformer import DAOnlyModel, DATransformer
    from model_latlon.da_encoders import PointObPrEncoder, MicrowaveEncoder

    model = DAOnlyModel(
        conf,
        anl_encoders=[],
        obs_encoders=[MicrowaveEncoder(mesh_1bamua, conf)],
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



@launch(nodes={'halfmoon': 1} ,port=29501, start_method="spawn",clear_cache=False)#,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Dec31_datest():
    #config.prefix = '_penguins'
    config.nope = True
    config.gpus = '0-5'
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
    timesteps = [6,30]
    #timesteps = [6]

    from model_latlon.da_encoders import MicrowaveData, BalloonData
    mesh_1bamua = MicrowaveData("1bamua") # atms,1bamua
    #igra_mesh = BalloonData()

    data = WeatherDataset(DataConfig(inputs=[imesh1, imesh2, mesh_1bamua], outputs=[omesh],
                                            timesteps=timesteps,
                                            only_at_z=[0,6,12,18],
                                            requested_dates = tdates
                                            ))
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
    config.optim = 'adam'
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
    w.run()


@launch(nodes={'glass': 6} ,port=29501, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Dec16_realtimesucks():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
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
    timesteps = [0,6,18,24]
    data = WeatherDataset(DataConfig(inputs=[imesh1, imesh2], outputs=[omesh],
                                            timesteps=timesteps,
                                            only_at_z=[0,6,12,18],
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    import evals.package_neo as pn

    model = pn.get_joanrealtimesucks()
    for param in model.parameters():
        param.requires_grad = False

    for param in model.encoders.parameters():
        param.requires_grad = True

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
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()




@launch(nodes={'bimini': 6} ,port=29501, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Dec16_joanbikes():
    #config.prefix = '_penguins'
    config.nope = False
    config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp', '143_cp', '201_mx2t', '202_mn2t']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [0,6,24,48]#,36]
    timesteps = [0,6,12,18,24,30,36,42,48,60,72,78,96,120]
    timesteps = [1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,19,20,21,22,23]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=timesteps,
                                            only_at_z=[0,6,12,18],
                                            random_timestep_subset=4,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    import evals.package_neo as pn

    model = pn.get_joanbikessucks()
    for param in model.parameters():
        param.requires_grad = False

    for param in model.processors['1'].parameters():
        param.requires_grad = True


    config.strict_load = True
    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 10
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

    config.lr_sched.cosine_period = 25_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.1e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 1000_000
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()




@launch(nodes={'barceloneta': 6} ,port=29502, start_method="spawn",clear_cache=False, kill_nvidia=False)#,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Dec16_rotary_ft():
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

    #imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    #omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    #timesteps = [0,6,24,48]#,36]
    timesteps = [0,6,12,18,24,30,36,42,48,60,72,78,96,120]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=timesteps,
                                            random_timestep_subset=5,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig, ResConvEncoder, ResConvDecoder
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1280
    #conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    conf.affine = True
    conf.dims_per_head = 40
    conf.pr_depth = 10
    conf.encdec_tr_depth = 4
    conf.oldpr = True
    conf.tr_embedding = 'rotary'
    conf.update()

    encoder = ResConvEncoder(imesh,conf)
    decoder = ResConvDecoder(omesh,conf)

    model = ForecastModel(imesh,conf, encoders=[encoder], decoders=[decoder])

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
    config.shampoo.dim = 2048
    config.reset_optimizer = True
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    config.lr_sched.restart_warmup_end_step = 500

    config.reset_steps_on_resume = True#False#True

    config.lr_sched.cosine_period = 20_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.025e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.strict_load = False
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 100000
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()



@launch(nodes={'bimini': 6} ,port=29501, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Dec6_joanlatentsucks():
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
    timesteps = [0,6,12,18,24,30,36,42,48,60,72,78,96,120]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=timesteps,
                                            only_at_z=[0,6,12,18],
                                            random_timestep_subset=5,
                                            requested_dates = tdates
                                            ))
    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    conf.affine = True
    conf.encdec_tr_depth = 2
    conf.oldpr = True
    encoder = ResConvEncoder(imesh,conf)
    decoder = ResConvDecoder(omesh,conf)

    model = ForecastModel(imesh,conf, encoders=[encoder], decoders=[decoder])
    print(model)

    config.strict_load = False # TODO: undo this when trying a different model!!!! only the const_data stuff should be there i think
    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 10
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo'
    #config.optim = 'adam'
    config.shampoo.dim = 4096
    config.reset_optimizer = True
    config.reset_steps_on_resume = True#False#True
    config.save_optimizer = True
    #config.lr_sched.cosine_period = 22_000
    #config.lr_sched.warmup_end_step = 1000
    #config.lr_sched.lr = 2e-4
    #config.lr_sched.restart_warmup_end_step = 500

    config.lr_sched.cosine_period = 16_000
    config.lr_sched.warmup_end_step = 500
    config.lr_sched.lr = 0.05e-3
    config.lr_sched.max_dt_min = 24
    config.lr_sched.max_dt_max = 48
    config.lr_sched.steps_till_max_dt_max = 30_000
    config.lr_sched.num_random_subset_min = 3
    config.lr_sched.num_random_subset_max = 5
    config.lr_sched.steps_till_num_random_subset_max = 30_000
    config.save_imgs_every = 1_000
    config.weight_eps = 0.02
    #config.profile = not config.nope
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()




@launch(nodes={'stinson': 4} ,port=29506, start_method="spawn",clear_cache=False, kill_nvidia=False, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Nov29_liere():
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

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1280
    #conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    conf.affine = True
    conf.pr_depth = 10
    # hi john. when setting up the run, do conf.dims_per_head = 40 (this defaults to 32 heads total, so 1280/32=40 per head)
    conf.encdec_tr_depth = 4
    conf.oldpr = True
    conf.tr_embedding = 'liere'
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
    config.shampoo.dim = 1024
    config.reset_optimizer = True
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




@launch(nodes={'barceloneta': 5, 'baga.fast': 5} ,port=29501, start_method="spawn",clear_cache=False, kill_nvidia=False)#,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Nov29_rotary_resume():
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

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1280
    #conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    conf.affine = True
    conf.pr_depth = 10
    conf.encdec_tr_depth = 4
    conf.oldpr = True
    conf.tr_embedding = 'rotary'
    model = ForecastModel(imesh,conf)
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



#@launch(nodes={'stinson': 4} ,port=29505, start_method="spawn",clear_cache=False)#,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
@launch(nodes={'barceloneta': 3} ,port=29504, start_method="spawn",clear_cache=False, kill_nvidia=False)#,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Nov29_baseline():
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

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(imesh)
    conf.latent_size = 1280
    #conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    conf.affine = True
    conf.pr_depth = 10
    conf.encdec_tr_depth = 4
    conf.oldpr = True
    conf.tr_embedding = 'sincos'
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




@launch(nodes={'halfmoon': 1} ,port=29505, start_method="spawn",clear_cache=False,kill_nvidia=False)#, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Nov18_sanitycheck():
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
    conf.latent_size = 768
    conf.pr_dims = [48, 192, 256]
    conf.affine = True
    conf.encdec_tr_depth = 2
    conf.oldpr = False
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
    config.optim = 'adam'
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


@launch(nodes={'barceloneta': 6} ,port=29505, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Nov22_joansucks_newpr():
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
    conf.pr_dims = [48, 192, 512]
    conf.affine = True
    conf.encdec_tr_depth = 2
    conf.oldpr = False
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



@launch(nodes={'bimini': 6} ,port=29505, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Nov22_joansucks_oldpr():
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
    conf.pr_dims = [48, 192, 512]
    conf.affine = True
    conf.encdec_tr_depth = 2
    conf.oldpr = True
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



@launch(nodes={'miramar': 6} ,port=29504, start_method="spawn",clear_cache=False,kill_nvidia=True, zulip=True, ping='@**Joan Creus-Costa**',validate=False)
def Nov12_oldpr():
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

@launch(nodes={'glass': 3} ,port=29504, start_method="spawn",clear_cache=False)#,kill_nvidia=False, zulip=True, ping='@**John Dean**',validate=False)
#@launch(ddp=0,start_method='spawn')
def full_mini_333():
    config.prefix = '_diffusion'
    #config.nope = True
    config.gpus = '0'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

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
