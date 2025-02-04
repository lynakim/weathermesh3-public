from train import *


args = parse_args()

if args.activity == 'Oct3-Longcos':
    args.resume = '_Oct3-Longcos_'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh)
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


if args.activity == 'Oct5':
    args.resume = '_-Oct5-Longcos-4xgpu-_'
    args.gpus = '0-3'
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh)
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


if args.activity == 'Oct5-222conv2':
    args.name = 'Oct5-222conv2'
    #args.resume = '_Oct5-222conv_'
    args.gpus = '0-1'
    args.nope = False
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    model = ForecastStepSwin3D(mesh)
    model.output_deltas = True
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.DH = 12
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 60_000
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
