from utils import *


def da_compare(trainer,x_gpu,dts,yt_gpus,rms_with_da):
    dh = trainer.model.encoders[0].mesh.hour_offset

    #NOTE: the way the dh is added and then removed is hacky. Ideally this should be fixed in the process.py file
    # when making the todo list, it should look at the input mesh. But it works for now.
    with torch.autocast(enabled=bool(trainer.conf.HALF), device_type='cuda', dtype=torch.float16):
        y_gpus = trainer.active_model.forcast_only(x_gpu,[dt+dh for dt in dts])
    del y_gpus['latent_l2']
    y_gpus = {dt: y_gpus[dt+dh] for dt in dts}
    
    Nt = len(dts)
    Nd = lambda i: len(y_gpus[dts[i]])

    def get_rms(i,j):
        dt = dts[i]
        return trainer.model.decoders[j].compute_errors(y_gpus[dt][j], yt_gpus[i][j], trainer=trainer)
    rms_no_da = { dts[i]: { j: get_rms(i, j) for j in range(Nd(i)) } for i in range(Nt)}

    ratios = { dts[i]: { j: {k: rms_with_da[dts[i]][j][k] / rms_no_da[dts[i]][j][k] for k in rms_with_da[dts[i]][j].keys()} for j in range(Nd(i)) } for i in range(Nt)}

    decoder = trainer.model.decoders[0]
    writer = trainer.writer

    for i in range(Nt):
        dt = dts[i]
        for var_name in decoder.mesh.pressure_vars + decoder.mesh.sfc_vars:
            writer.add_scalar(f"DA_Ratio_{dt}/" + var_name, ratios[dt][0][var_name], trainer.state.n_step)
        for var_name in decoder.mesh.pressure_vars:
            name500 = var_name + "_500"
            writer.add_scalar(f"DA_Ratio_{dt}/" + name500, ratios[dt][0][name500], trainer.state.n_step)
        print(f"DA Ratio {dt} z_500: {ratios[dt][0]['129_z_500']:.2f} | no_da {rms_no_da[dt][0]['129_z_500']:.2f}")

    
    
