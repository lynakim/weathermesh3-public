from train import *


if __name__ == '__main__':
    args = parse_args()

    if args.activity == '3d1':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin3D(mesh)
        model.output_deltas = True
        w = WeatherTrainer(args,mesh,model)
        w.audit = True
        w.conf.dates = get_dates((D(2015, 1, 1),D(2015, 1,2))) 
        w.conf.shuffle = False
        w.conf.only_at_z = [12]
        print(model)
        w.conf.log_every = 10
        w.run()


    if args.activity == 'overfit3d':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin3D(mesh)
        model.output_deltas = False
        w = WeatherTrainer(args,mesh,model)
        w.preserved_conf.dates = get_dates((D(2015, 1, 1),D(2015, 1,16))) 
        w.preserved_conf.N_epochs = 1000000000 
        w.conf.dates = get_dates((D(2015, 1, 1),D(2015, 1,4))) 
        w.conf.only_at_z = [12]
        w.conf.log_every = 10
        w.conf.DH = 3
        w.conf.lr_sched.lr *= 3*10
        w.run()

    if args.activity == 'overfit2d':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin(mesh)
        w = WeatherTrainer(args,mesh,model)
        w.conf.N_epochs = 1000000000
        w.conf.dates = get_dates((D(2015, 1, 1),D(2015, 1,4))) 
        w.conf.shuffle = False
        w.conf.only_at_z = [12]
        w.conf.DH = 3
        w.conf.lr_sched.lr *= 10
        w.conf.HALF = 0
        w.conf.log_every = 20
        w.run()


    if args.activity == '3d':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin3D(mesh)
        model.output_deltas = False
        w = WeatherTrainer(args,mesh,model)
        w.preserved_conf.N_epochs = 1000000000 
        w.preserved_conf.only_at_z = [12]
        w.preserved_conf.dates = get_dates((D(2007, 1, 1),D(2017, 12,1)))
        w.preserved_conf.log_every = 20
        w.preserved_conf.DH = 12
        w.preserved_conf.lr_sched = SimpleNamespace()
        w.preserved_conf.lr_sched.lr = 1e-4 / 2
        w.preserved_conf.lr_sched.cosine_en = 0 
        w.run()

    if args.activity == '3dcos':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin3D(mesh)
        model.output_deltas = True
        w = WeatherTrainer(args,mesh,model)
        w.preserved_conf.N_epochs = 1000000000 
        w.preserved_conf.only_at_z = [12]
        w.preserved_conf.dates = get_dates((D(2007, 1, 1),D(2017, 12,1)))
        w.preserved_conf.log_every = 20
        w.preserved_conf.DH = 12
        w.preserved_conf.lr_sched = SimpleNamespace()
        w.preserved_conf.lr_sched.lr = 5e-5#1e-3 / 2**0.5 
        w.preserved_conf.lr_sched.cosine_en = 1
        w.preserved_conf.lr_sched.step_offset = -21_100*2
        w.preserved_conf.lr_sched.cosine_period = 40_000
        w.run()
    
    if args.activity == 'convonly':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepConvOnly(mesh)
        model.output_deltas = True 
        w = WeatherTrainer(args,mesh,model)
        w.preserved_conf.N_epochs = 1000000000 
        w.preserved_conf.dates = get_dates((D(2015, 1, 1),D(2015, 2,1))) 
        #w.preserved_conf.only_at_z = [12]
        w.preserved_conf.log_every = 20
        w.preserved_conf.DH = 0
        w.preserved_conf.lr_sched = SimpleNamespace()
        w.preserved_conf.lr_sched.lr = 1e-4 / 3
        w.preserved_conf.lr_sched.cosine_en = 0 
        w.run()