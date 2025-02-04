from launch import * 
from train import *


@launch(nodes={'stinson':4})
def Nov2_wb2():
    args.gpus = '0-3'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
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
    w.preserved_conf.l2_loss = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.1e-3 
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 5e-6
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 16
    w.preserved_conf.validate_every = 200
    w.run()

@launch(nodes={'rockaway.fast': 6, 'barceloneta':6})
def Nov6_48h():
    args.gpus = '0-5'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
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
    w.preserved_conf.output_DH = 48
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.05e-3  
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
    w.preserved_conf.validate_N = 24
    w.preserved_conf.validate_every = 250
    w.preserved_conf.save_every = 400
    w.run()


@launch(nodes={'rockaway.fast': 6, 'barceloneta':6})
def Nov5_double():
    args.gpus = '0-5'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
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
    w.preserved_conf.output_DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.05e-3  
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
    w.preserved_conf.validate_N = 24
    w.preserved_conf.validate_every = 250
    w.preserved_conf.save_every = 400
    w.run()

@launch(nodes={'halfmoon.fast': 4, 'stinson':4})
def Nov6_l2_wb():
    args.gpus = '0-3'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
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
    w.preserved_conf.l2_loss = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.15e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 5e-6
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 24
    w.preserved_conf.validate_every = 250
    w.preserved_conf.save_every = 400
    w.run()

@launch(nodes={'halfmoon.fast':4,'stinson':4})
def Nov8_6hr():
    args.gpus = '0-3'
    #args.nope = True
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=16, drop_path=0, checkpoint_every=1)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 6
    w.preserved_conf.output_DH = 6
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
    w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 24
    w.preserved_conf.validate_every = 250
    w.preserved_conf.save_every = 400
    w.run()


@launch(nodes={'halfmoon.fast':4,'stinson':4})
def Nov10_l1cont():
    args.gpus = '0-3'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.reset_optimizer = False
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
    w.preserved_conf.lr_sched.warmup_end_step = 5_000
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 1_000_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.adam.betas = (0.9, 0.999)
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 24
    w.preserved_conf.validate_every = 250
    w.preserved_conf.save_every = 400
    w.run()


@launch(nodes={'martins':1})
def Nov13_perftest():
    args.gpus = '0'
    args.nope = True
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=512, depth=48, drop_path=0, skip_every=3, window_size=(2,3,6))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 50
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.2e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = None
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.adam.betas = (0.9, 0.999)
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 24
    w.preserved_conf.validate_every = 250
    w.preserved_conf.save_every = 200
    w.run()


@launch(nodes={'rockaway':6})
def Nov13_longlong2():
    #os.envirno['CUDA_VISIBLE_DEVICES'] = '0,1,2,4,5'
    """
    CUDA_VISIBLE_DEVICES=1,2,4,5 python3 runs/Oct27.py Nov13_longlong2
    so far
    01245
    0245
    1245 <----- immediately bad
    0234
    0235
    0134
    """
    #args.gpus = '1,2,4,5'
    args.gpus = '0-5'
    args.nope = True
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=512, depth=72, drop_path=0, skip_every=None, window_size=(2,3,6))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 50
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.3e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = None
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.adam.betas = (0.9, 0.999)
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 12
    w.preserved_conf.validate_every = 250
    w.preserved_conf.save_every = 500
    w.run()


@launch(nodes={'stinson':4})
def Nov13_longlongman():
    args.gpus = '0-5'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=512, depth=72, drop_path=0, skip_every=None, window_size=(2,3,6))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 50
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.GRAD_ACCUM = 2
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.075e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = None
    w.preserved_conf.lr_sched.div_factor = 3
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 300_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.
    w.preserved_conf.adam.betas = (0.9, 0.999)
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 12
    w.preserved_conf.validate_every = 250
    w.preserved_conf.save_every = 500
    w.run()


@launch(nodes={'halfmoon.fast': 4, 'barceloneta':6}, port=29503)
def Nov14_patience2():
    args.gpus = '0-5'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.GRAD_ACCUM = 4
    w.preserved_conf.log_every = 50
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.75e-4  * 2.1/5.75
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 0.5e-5
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 10_000
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.adam.betas = (0.9, 0.999)
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 10
    w.preserved_conf.validate_every = 100
    w.preserved_conf.save_every = 100
    w.run()



@launch(nodes={'halfmoon.fast': 4, 'barceloneta':6})
def Nov14_patience():
    args.gpus = '0-5'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.reset_optimizer = False
    w.preserved_conf.log_every = 50
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.5e-4 
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 0.8e-5
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 7_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 200_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.adam.betas = (0.9, 0.999)
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 10
    w.preserved_conf.validate_every = 250
    w.preserved_conf.save_every = 200
    w.run()



@launch(nodes={'halfmoon.fast': 4, 'barceloneta':6})
def Nov14_cont10():
    args.gpus = '0-5'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 50
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.2e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 0.5e-5
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.adam.betas = (0.9, 0.999)
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1997, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 10
    w.preserved_conf.validate_every = 250
    w.preserved_conf.save_every = 200
    w.run()


@launch(nodes={'barceloneta':6})
def Nov7_l1_wb():
    args.gpus = '0-5'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 50
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.2e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 0.5e-5
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.adam.betas = (0.9, 0.999)
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 24
    w.preserved_conf.validate_every = 250
    w.preserved_conf.save_every = 200
    w.run()

   
@launch(nodes={'halfmoon.fast':4,'stinson':4})
def Nov11_skip():
    args.gpus = '0-3'
    #args.nope = True
    #args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, drop_path=0, skip_every=6)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 25
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = False
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.15e-3  
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 5e-6
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 1000_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.adam.betas = (0.9, 0.999)
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 24
    w.preserved_conf.validate_every = 300
    w.preserved_conf.save_every = 400
    w.run()


@launch(nodes={'stinson':4})
def Nov2_wb():
    args.gpus = '0-3'
    #args.nope = True
    args.resume = "_"+args.activity.replace("_","-")+"_"
    mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
    model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=28, drop_path=0)
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(args,mesh,model)
    w.preserved_conf.log_every = 20
    w.preserved_conf.actually_scale_gradients = True
    w.preserved_conf.clamp = 12
    w.preserved_conf.only_at_z = None
    w.preserved_conf.DH = 24
    w.preserved_conf.l2_loss = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.lr = 0.1e-3 
    w.preserved_conf.lr_sched.cosine_en = 1
    w.preserved_conf.lr_sched.cosine_bottom = 1.5e-5
    w.preserved_conf.lr_sched.div_factor = 4
    w.preserved_conf.lr_sched.warmup_end_step = 2_500
    #w.preserved_conf.lr_sched.step_offset = 205_000
    w.preserved_conf.lr_sched.cosine_period = 800_000
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.weight_decay = 0.003
    w.preserved_conf.HALF = 1
    w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.val_dates = get_dates((D(2018, 1, 1),D(2018, 12, 30)))
    w.preserved_conf.validate_N = 16
    w.preserved_conf.validate_every = 200
    w.run()


@launch(nodes={'halfmoon.fast':4,'stinson':4})
def Oct31_spooky():
    args.gpus = '0-3'
    args.nope = True
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
    w.preserved_conf.validate_N = 24
    w.run()

if __name__ == '__main__':
    run(locals().values())
