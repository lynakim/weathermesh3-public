import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import *


args = parse_args()

if args.activity == 'profiletest':
    #args.resume = '_Oct7-CosRep_'
    args.name= 'profiletest'
    args.gpus = '2'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4))
    model.output_deltas = True
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    #w.preserved_conf.lr_sched.step_offset= 21_000
    w.preserved_conf.lr_sched.cosine_period = 40_000
    w.run()


if args.activity == 'Oct7-CosRep-hugehalf':
    #args.resume = '_Oct7-CosRep_'
    args.name= 'Oct7-CosRep-hugehalf'
    args.gpus = '0-1'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), FLASH=False, conv_dim=1024, dims_per_head=32)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    #w.preserved_conf.lr_sched.step_offset= 21_000
    w.preserved_conf.lr_sched.cosine_period = 40_000
    w.preserved_conf.HALF = True
    w.run()

if args.activity == 'Oct7-CosRep-hugehalf-adam':
    #args.resume = '_Oct7-CosRep_'
    args.name= 'Oct7-CosRep-hugehalf-adam'
    args.gpus = '2-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), FLASH=False, conv_dim=1024, dims_per_head=32)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 4e-5
    w.preserved_conf.lr_sched.cosine_en = 1
    #w.preserved_conf.lr_sched.step_offset= 21_000
    w.preserved_conf.lr_sched.cosine_period = 100_000
    w.preserved_conf.optim = torch.optim.Adam
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.0
    w.preserved_conf.adam.betas = (0.9, 0.999)
    w.preserved_conf.HALF = True
    w.run()


if args.activity == 'Oct7-CosRep-hugehalf-halfheads':
    #args.resume = '_Oct7-CosRep_'
    args.name= 'Oct7-CosRep-hugehalf-halfheads'
    args.gpus = '2-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), FLASH=False, conv_dim=1024, dims_per_head=64)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    #w.preserved_conf.lr_sched.step_offset= 21_000
    w.preserved_conf.lr_sched.cosine_period = 40_000
    w.preserved_conf.HALF = True
    w.run()




if args.activity == 'Oct7-CosRep':
    #args.resume = '_Oct7-CosRep_'
    args.name= 'Oct7-CosRep'
    args.gpus = '0-1'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=512)
    model.output_deltas = True
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    #w.preserved_conf.lr_sched.step_offset= 21_000
    w.preserved_conf.lr_sched.cosine_period = 40_000
    w.run()

if args.activity == 'Oct7-CosRep-fp16':
    #args.resume = '_Oct7-CosRep_'
    args.name= 'Oct7-CosRep-fp16'
    args.gpus = '2-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=512)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    #w.preserved_conf.lr_sched.step_offset= 21_000
    w.preserved_conf.lr_sched.cosine_period = 40_000
    w.preserved_conf.HALF = 1
    w.run()

if args.activity == 'Oct9-CosRep-fp16-mae':
    #args.resume = '_Oct7-CosRep_'
    args.name= 'Oct9-CosRep-fp16-mae'
    args.gpus = '0-1'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=512)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    #w.preserved_conf.lr_sched.step_offset= 21_000
    w.preserved_conf.lr_sched.cosine_period = 40_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.005
    w.preserved_conf.HALF = 1
    w.run()


if args.activity == 'Oct10-hopium-hm-24h':
    args.name= args.activity
    #args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=768, depth=10)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    #w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.0025
    w.preserved_conf.HALF = 1
    w.run()


if args.activity == 'Oct10-fullhopium-3xlr-24h':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=768, depth=10)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1.5e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.0025
    w.preserved_conf.HALF = 1
    w.run()


if args.activity == 'Oct10-fullhopium-beta-24h':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=768, depth=10)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.5e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.0025
    w.preserved_conf.adam.betas = (0.9, 0.999)
    w.preserved_conf.HALF = 1
    #w.conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    #w.conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


if args.activity == 'Oct10-fullhopium-0.25-down':
    args.name= args.activity
    #args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=10, drop_path=True)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.5e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.0025
    w.preserved_conf.HALF = 1
    w.conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


if args.activity == 'Oct10-fullhopium-gradscale':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=1024, depth=10, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = None
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()



if args.activity == 'Oct10-fullhopium-longerdrop':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=1024, depth=10, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 1e-5
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()

if args.activity == 'Nov4-layers-neo':
    args.name= args.activity
    #args.resume = "_"+args.activity+"_"
    args.gpus = '0-5'
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.2e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 5e-6
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 18
    w.preserved_conf.validate_every = 500
    w.run()



if args.activity == 'Nov1-layers':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-5'
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.2e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 5e-6
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 18
    w.preserved_conf.validate_every = 500
    w.run()



if args.activity == 'Oct26-hres-contslow':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-5'
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=1024, depth=10, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.075e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 5e-6
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 18
    w.preserved_conf.validate_every = 500
    w.run()



if args.activity == 'Oct26-hres':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=1024, depth=10, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


if args.activity == 'Oct26-hresdrop':
    args.name= args.activity
    #args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=10, drop_path=True)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.5e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


if args.activity == 'Oct10-fullhopium-clampscaledrop':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-4'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=1024, depth=10, drop_path=True)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()


if args.activity == 'Oct10-fullhopium-clampscale':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=1024, depth=10, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.run()



if args.activity == 'Oct10-fullhopium-nodrop-24h':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=768, depth=10, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.5e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.0025
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1998, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(1997, 1, 1),D(1997, 12, 30)))
    w.run()




if args.activity == 'Oct10-fullhopium-drop-24h':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=768, depth=10, drop_path=True)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.5e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.0025
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1998, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(1997, 1, 1),D(1997, 12, 30)))
    w.run()


if args.activity == 'Oct10-fullhopium-hm-24h':
    #args.name= args.activity
    args.resume = "_"+args.activity+"_"
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=768, depth=10)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.5e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.0025
    w.preserved_conf.HALF = 1
    w.run()



if args.activity == 'Oct10-CosRep-fp16-maebig':
    #args.name= 'Oct10-CosRep-fp16-maebig'
    args.resume= '_Oct10-CosRep-fp16-maebig_'
    args.gpus = '0-1'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=1024)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 2e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.0025
    w.preserved_conf.HALF = 1
    w.run()




if args.activity == 'Oct9-CosRep-fp16-maelong':
    #args.resume = '_Oct7-CosRep_'
    args.resume = '_Oct9-CosRep-fp16-maelong_'
    args.name= 'Oct9-CosRep-fp16-maelong'
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=512)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 3.5e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 7_500
    w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.005
    w.preserved_conf.HALF = 1
    w.run()

if args.activity == 'Oct9-CosRep-fp16-maebiglong':
    #args.resume = '_Oct7-CosRep_'
    #args.resume = '_Oct9-CosRep-fp16-maelong_'
    args.name= 'Oct9-CosRep-fp16-maebiglong'
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=1024)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 7_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.005
    w.preserved_conf.HALF = 1
    w.run()




if args.activity == 'Oct7-CosRep-fp16-222':
    #args.resume = '_Oct7-CosRep_'
    args.name= 'Oct7-CosRep-fp16-222'
    args.resume = '_Oct7-CosRep-fp16-222_'
    args.gpus = '0-1'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,2,2), conv_dim=384)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0#0.001
    #w.preserved_conf.lr_sched.step_offset= 21_000
    w.preserved_conf.lr_sched.cosine_period = 40_000
    w.preserved_conf.HALF = 1
    w.run()



if args.activity == 'Oct5-242conv':
    #args.name = 'Oct5-242conv'
    args.resume = '_Oct5-242conv_'
    args.gpus = '0-3'
    #args.nope = True
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh,patch_size=(2,4,2))
    model.output_deltas = True
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 60_000
    w.run()


if args.activity == 'Oct11-Regtest':
    args.name = args.activity
    #args.nope = True
    #args.gpus = '0-1'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh,patch_size=(2,4,4))
    model.output_deltas = True
    w = WeatherTrainer(args,mesh,model)
    w.conf.DDP = True
    w.run()

if args.activity == 'Oct17-DDPtest':
    args.name = args.activity
    args.none = True
    #args.resume = "_"+args.activity+"_"
    #args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=768, depth=10)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 1.5e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.0025
    w.preserved_conf.HALF = 1
    w.run()

#@ddp(nnodes=2,nproc_per_node=4)
#def Oct17_Launcher_Test():
#    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
#    model = ForecastStepSwin3D(mesh, patch_size=(2,4,4), conv_dim=768, depth=10)
#    model.output_deltas = True
#    model.do_sub = False
#    w = WeatherTrainer(args,mesh,model)
