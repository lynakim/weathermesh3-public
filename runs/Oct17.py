from train import *
from launch import * 

@launch()
def Oct19_Ohp():
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
    w.run()


if __name__ == '__main__':
    run(locals().values())