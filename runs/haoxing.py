import sys
sys.path.append('/fast/haoxing/deep')
from runs.launch import *

if am_i_torchrun():
    from train.trainer import *

from train.trainer import * # <--- comment this back in if not using DDP
from torch.utils import checkpoint as torch_checkpoint

# set cuda visible devices if on stinson
if 'stinson' == os.uname()[1]:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

@launch(nodes={'singing': 4} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan16_pointdec_5kmgrid():
    config.nope = False
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        tr2d_hidden_dim=768,
        tr2d_depth=2,
        tr2d_num_heads=24,
        tr2d_window_size=(7,7),
        n_deconv_channels=256,
        inner_res=0.05,
        local_patch_size=0.5,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [0,6]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=512)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
    w.run()

@launch(nodes={'gold': 3} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan15_pointdec_10kmgrid():
    config.nope = False
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        tr2d_hidden_dim=768,
        tr2d_depth=2,
        tr2d_num_heads=24,
        tr2d_window_size=(7,7),
        deconv_kernel_size=(20,20),
        n_deconv_channels=256,
        margin=2,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [0,6]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=512)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 5
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo' #'adam'
    config.shampoo.dim = 4096
    config.shampoo.version = 'new'
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
    w.run()

@launch(nodes={'singing': 4} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan15_pointdec_centeredpatches():
    config.nope = False
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        tr2d_hidden_dim=1024,
        tr2d_depth=2,
        tr2d_num_heads=32,
        tr2d_window_size=(7,7),
        n_deconv_channels=256,
        margin=2,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [0,6]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=512)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
    w.run()

#@launch(ddp=0, start_method='spawn')
@launch(nodes={'singing': 4} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan14_pointdec_posemb():
    config.nope = False
    config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        tr2d_hidden_dim=1024,
        tr2d_depth=2,
        tr2d_num_heads=32,
        tr2d_window_size=(7,7),
        n_deconv_channels=256,
        margin=2,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [0,6]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=512)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
    w.run()

@launch(ddp=0, start_method='spawn')
def Jan14_pointdec_debug():
    config.nope = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        tr2d_hidden_dim=768,
        tr2d_depth=2,
        tr2d_num_heads=24,
        tr2d_window_size=(7,7),
        n_deconv_channels=256,
        inner_res=0.05,
        local_patch_size=0.5,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    # tdates = get_dates([(D(2010, 12, 5), D(2010, 12, 7))])
    # tdates = get_dates([(D(1990, 8, 24), D(1990, 8, 31))])
    # tdates = get_dates([(D(2013, 6, 17), D(2013, 6, 22))])
    #dates = get_dates([(D(2019, 12, 21), D(2019, 12, 22))])
    timesteps = [0, 6]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=512)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
    config.reset_optimizer = False
    config.save_optimizer = False
    config.lr_sched = LRScheduleConfig(
        schedule_dts=False,
        cosine_period=200_000,
        warmup_end_step=1_000,
        restart_warmup_end_step=500,
        lr=0.7e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()




#@launch(ddp=0, start_method='spawn')
@launch(nodes={'singing': 4} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan13_pointdec_mod2():
    config.nope = False
    config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        hidden_dim=1024,
        tr2_depth=4,
        margin=2,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6, 24]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=512)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
    config.loss_consts_override = {6:1, 24:1}
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(nodes={'singing': 4} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
@launch(ddp=0, start_method='spawn')
def Jan13_pointdec_tr2d2_lowlr():
    config.nope = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        hidden_dim=1024,
        tr2_depth=2,
        margin=2,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6, 24]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=512)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
    config.reset_optimizer = False
    config.save_optimizer = False
    config.lr_sched = LRScheduleConfig(
        schedule_dts=False,
        cosine_period=200_000,
        warmup_end_step=1_000,
        restart_warmup_end_step=500,
        lr=0.7e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'singing': 4} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan13_pointdec_mergemaintest():
    config.nope = False
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        hidden_dim=1024,
        tr2_depth=1,
        margin=2,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6, 24]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=512)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 5
    config.log_step_every = 1
    config.save_every = 10
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo' #'adam'
    config.shampoo.dim = 4096
    config.shampoo.version = 'old'
    config.reset_optimizer = False
    config.save_optimizer = False
    config.lr_sched = LRScheduleConfig(
        schedule_dts=False,
        cosine_period=100_000,
        warmup_end_step=1000,
        restart_warmup_end_step=500,
        lr=1e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0, start_method='spawn')
@launch(nodes={'singing': 4} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan10_pointdec_fixidx_b512():
    config.nope = False
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        hidden_dim=1024,
        margin=2,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6, 24]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=512)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
    config.reset_optimizer = False
    config.save_optimizer = False
    config.lr_sched = LRScheduleConfig(
        schedule_dts=False,
        cosine_period=100_000,
        warmup_end_step=1000,
        restart_warmup_end_step=500,
        lr=1e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'gold': 3} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan9_pointdec_morecomplex_smallbatch():
    config.nope = False
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        hidden_dim=1024,
        margin=2,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6,24]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=768)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 5
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo' #'adam'
    config.shampoo.dim = 4096
    config.shampoo.version = 'new'
    config.reset_optimizer = False
    config.save_optimizer = False
    config.lr_sched = LRScheduleConfig(
        schedule_dts=False,
        cosine_period=100_000,
        warmup_end_step=1000,
        restart_warmup_end_step=500,
        lr=1e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    config.decoder_loss_weights = [1, 0.5]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0, start_method='spawn')
@launch(nodes={'singing': 4} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan9_pointdec_morecomplex():
    config.nope = False
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        hidden_dim=1024,
        margin=2,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6,24]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=2048)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
    config.reset_optimizer = False
    config.save_optimizer = False
    config.lr_sched = LRScheduleConfig(
        schedule_dts=False,
        cosine_period=100_000,
        warmup_end_step=1000,
        restart_warmup_end_step=500,
        lr=1e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    config.decoder_loss_weights = [1, 0.5]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0, start_method='spawn')
@launch(nodes={'singing': 4} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan9_pointdec_moreinterm():
    config.nope = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        hidden_dim=1024,
        margin=1,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6,24]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=2048)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
        cosine_period=100_000,
        warmup_end_step=1000,
        restart_warmup_end_step=500,
        lr=1e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    config.decoder_loss_weights = [1, 0.5]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

@launch(nodes={'gold': 3} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan8_pointdec_mlpout_fixed2():
    config.nope = False
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        hidden_dim=1024,
        margin=1,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6,24]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=3000)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 5
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo' #'adam'
    config.shampoo.dim = 4096
    config.shampoo.version = 'new'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched = LRScheduleConfig(
        schedule_dts=False,
        cosine_period=100_000,
        warmup_end_step=1000,
        restart_warmup_end_step=500,
        lr=1e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    config.decoder_loss_weights = [1, 0.5]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


#@launch(ddp=0, start_method='spawn')
@launch(nodes={'gold': 3} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Jan8_pointdec_mlpout_fixed():
    config.nope = False
    config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        hidden_dim=1024,
        margin=1,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6,24]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=1024)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 5
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo' #'adam'
    config.shampoo.dim = 4096
    config.shampoo.version = 'new'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched = LRScheduleConfig(
        schedule_dts=False,
        cosine_period=100_000,
        warmup_end_step=1000,
        restart_warmup_end_step=500,
        lr=1e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    config.decoder_loss_weights = [1, 0.5]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()


#@launch(ddp=0, start_method='spawn')
@launch(nodes={'singing': 4} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=False)#, zulip=True, ping='@**Haoxing Du**')
def Jan7_pointdec_refactored():
    config.nope = False
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    from datasets import AnalysisDataset, StationDataset
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    station_mesh = meshes.StationData()
    point_decoder = PointDecoder(
        station_mesh,
        conf,
        hidden_dim=1024,
        dims_per_head=32,
        n_point_vars=8,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6,24]

    data = WeatherDataset(DataConfig(
        inputs=[AnalysisDataset(imesh)],
        outputs=[
            AnalysisDataset(omesh), 
            StationDataset(station_mesh, batch_size=192)
        ],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
    config.reset_optimizer = False
    config.save_optimizer = False
    config.lr_sched = LRScheduleConfig(
        schedule_dts=False,
        cosine_period=100_000,
        warmup_end_step=1000,
        restart_warmup_end_step=500,
        lr=1e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    config.decoder_loss_weights = [1, 0.5]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(nodes={'gold': 2} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=True, zulip=True, ping='@**Haoxing Du**')
#@launch(ddp=0, start_method='spawn')
@launch(nodes={'barceloneta': 6} ,port=29524, start_method="spawn", clear_cache=False)#, kill_nvidia=True, zulip=True, ping='@**Haoxing Du**')
def Jan2_pointdecoderlonger():
    config.nope = False
    config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    point_decoder = PointDecoder(
        era5_decoder.mesh, # TODO: not the right mesh but also not used
        conf,
        hidden_dim=1024,
        dims_per_head=32,
        n_point_vars=8,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6,24]

    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    data = WeatherDataset(DataConfig(
        inputs=[imesh],
        outputs=[omesh],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
        cosine_period=100_000,
        warmup_end_step=1000,
        restart_warmup_end_step=500,
        lr=1e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1, 24:1}
    config.decoder_loss_weights = [1, 0.5]
    config.use_point_dataset = True
    config.point_batch_size = 1
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(nodes={'gold': 2} ,port=29524, start_method="spawn", clear_cache=True, kill_nvidia=True, zulip=True, ping='@**Haoxing Du**')
#@launch(nodes={'barceloneta': 6} ,port=29524, start_method="spawn", clear_cache=False, kill_nvidia=True, zulip=True, ping='@**Haoxing Du**')
@launch(ddp=0, start_method='spawn')
def Dec31_pointdecodertest():
    config.nope = True
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    from evals.package_neo import get_joansucks
    model = get_joansucks()
    for p in model.parameters():
        p.requires_grad = False
    
    from model_latlon.decoder import PointDecoder
    era5_decoder = model.decoders[0]
    conf = model.config
    conf.checkpoint_type = 'torch'
    point_decoder = PointDecoder(
        era5_decoder.mesh, # TODO: not the right mesh but also not used
        conf,
        hidden_dim=1024,
        dims_per_head=32,
        n_point_vars=8,
    )
    model.decoders = nn.ModuleList([era5_decoder, point_decoder])

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    timesteps = [6,24]

    imesh = model.config.inputs[0]
    omesh = model.config.outputs[0]
    data = WeatherDataset(DataConfig(
        inputs=[imesh],
        outputs=[omesh],
        timesteps=timesteps,
        requested_dates=tdates,
    ))

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
        cosine_period=22_000,
        warmup_end_step=1000,
        restart_warmup_end_step=500,
        lr=1e-4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1,24:1}
    config.decoder_loss_weights = [1, 0.5]
    config.use_point_dataset = True
    config.point_batch_size = 1
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0,start_method='spawn')
@launch(nodes={'gold': 2} ,port=29524, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Dec13_hres9km_new16levs():
    config.nope = False
    #   config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_ecm1)
    mesh2 = meshes.LatLonGrid(
        source='hres9km-25',
        extra_sfc_vars=[],
        input_levels=levels_ncarhres,
        levels=levels_ecm1,
        resolution=0.1,
    )

    timesteps = [6,24]
    data = WeatherDataset(DataConfig(
        inputs=[mesh1], 
        outputs=[mesh1, mesh2],
        timesteps=timesteps,
        requested_dates=tdates
    ))

    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvEncoder
    from model_latlon.decoder import SimpleConvDecoder, SimpleConvDecoder9km
    conf = ForecastModelConfig(
        inputs=data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth=4,
        latent_size=896,
        window_size=(3,5,7),
        patch_size=(4,8,8),
        patch_size_9km=(4,20,20),
    )
    model = ForecastModel(
        mesh1,
        conf,
        encoders=[SimpleConvEncoder(mesh1, conf)],
        decoders=[
            SimpleConvDecoder(mesh1, conf),
            SimpleConvDecoder9km(mesh2, conf)
        ]
    )
    print(model)

    print_total_params(model)

    config.diffusion = False
    config.ignore_train_safegaurd = True
    config.log_every = 25
    config.log_step_every = 1
    config.save_every = 100
    config.latent_l2 = 1e-4
    config.HALF = True
    config.optim = 'shampoo' #'adam'
    config.shampoo.dim = 4096
    config.shampoo.version = 'new'
    config.reset_optimizer = True
    config.save_optimizer = False
    config.lr_sched = LRScheduleConfig(
        schedule_dts=False,
        cosine_period=22_000,
        warmup_end_step=1000,
        restart_warmup_end_step=500,
        lr=2e-4,
        #max_dt_min=24,
        #max_dt_max=144,
        #steps_till_max_dt_max=15_000,
        #num_random_subset_min=4,
        #num_random_subset_max=4,
    )
    config.weight_eps = 0.02
    config.loss_consts_override = {6:1,24:1}
    config.decoder_loss_weights = [1, 0.5]
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.run()

#@launch(ddp=0,start_method='spawn')
@launch(nodes={'miramar': 6} ,port=29524, start_method="spawn",clear_cache=False,kill_nvidia=False)#, zulip=True, ping='@**Haoxing Du**')
def Dec6_hres9km_16levs():
    #config.prefix = '_penguins'
    config.nope = False
    #config.gpus = '0-5'
    config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_ecm1)
    mesh2 = meshes.LatLonGrid(
        source='hres9km-25',
        extra_sfc_vars=[],
        input_levels=levels_ncarhres,
        levels=levels_ecm1,
        resolution=0.1,
    )

    timesteps = [6,24]
    data = WeatherDataset(DataConfig(inputs=[mesh1], outputs=[mesh1, mesh2],
                                            timesteps=timesteps,
                                            requested_dates = tdates))
    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvDecoder, SimpleConvDecoder9km
    conf = ForecastModelConfig(
        inputs=[mesh1],
        outputs=[mesh1, mesh2],
        encdec_tr_depth=4,
        oldenc=True,
        latent_size=512,
        window_size=(3,5,7),
        simple_conv_kernel_size=(4,8,8),
        simple_conv_9km_kernel_size=(4,20,20),
        decoder_types=[SimpleConvDecoder, SimpleConvDecoder9km],
        decoder_configs=[
            SimpleNamespace(), 
            SimpleNamespace()]
    )
    model = ForecastModel(mesh1,conf)
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
@launch(nodes={'gold': 2} ,port=29524, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
def Dec5_hres9km_test():
    #config.prefix = '_penguins'
    config.nope = False
    #config.gpus = '0-5'
    #config.resume = "_"+config.activity.replace("_","-")+"_"

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(
        source='hres9km-25',
        extra_sfc_vars=[],
        input_levels=levels_ncarhres,
        levels=levels_ncarhres,
        resolution=0.1,
    )

    timesteps = [6,24]
    data = WeatherDataset(DataConfig(inputs=[mesh1], outputs=[mesh1, mesh2],
                                            timesteps=timesteps,
                                            requested_dates = tdates))
    config.lr_sched.schedule_dts = False

    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvDecoder, SimpleConvDecoder9km
    conf = ForecastModelConfig(
        inputs=[mesh1],
        outputs=[mesh1, mesh2],
        encdec_tr_depth=4,
        oldenc=True,
        latent_size=896,
        window_size=(3,5,7),
        decoder_types=[SimpleConvDecoder, SimpleConvDecoder9km],
        decoder_configs=[
            SimpleNamespace(), 
            SimpleNamespace()]
    )
    model = ForecastModel(mesh1,conf)
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
    config.shampoo.version = 'new'
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

#@launch(nodes={'stinson': 2} ,port=29524, start_method="spawn",clear_cache=False,kill_nvidia=False, zulip=True, ping='@**Haoxing Du**')
@launch(ddp=0,start_method='spawn')
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
 



if __name__ == '__main__':
    run(locals().values())
