import torch
import matplotlib.pyplot as plt
import numpy as np
import meshes
from utils import levels_medium, get_dates, levels_joank, D, get_date_str
from data import WeatherDataset, DataConfig
import os
from neovis import Neovis
from evals.package_neo import *
from eval import unnorm
from evals.evaluate_error import to_filename, save_instance
import copy

EVAL_2020_DATES = get_dates((D(2020, 1, 1),D(2020,12,31)))
NUM_DATES = 16
NUM_GEN = 16
NUM_STEPS = 32
TIME_STEPS = [6,12,24,36,48]
#TIME_STEPS = [6,24]
SAVE_IMGS = True
SAVE_PLOTS = True
device = 'cuda:0'
model = get_multigibbs()
save_path= f'/fast/to_srv/diffusion/Dec9-2-multigibbs' 


model.eval()
model.to(device)

mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=['logtp', '15_msnswrf', '45_tcc'], input_levels=levels_medium, levels=levels_joank)
mesh_sfc = copy.deepcopy(mesh)
mesh_sfc.full_varlist = mesh.sfc_vars
mesh_sfc.state_norm_stds = mesh.state_norm_stds[-mesh.n_sfc_vars:]
mesh_sfc.state_norm_means = mesh.state_norm_means[-mesh.n_sfc_vars:]
json = mesh_sfc.to_json(model_name=model.name)


tdates = get_dates([(D(2021, 1, 23), D(2021, 12, 28))])
tdates = EVAL_2020_DATES
timesteps = TIME_STEPS
data = WeatherDataset(DataConfig(inputs=[mesh], outputs=[mesh],
                                        timesteps=timesteps,
                                        requested_dates = tdates,
                                        only_at_z=[0,12],
                                        ))
data.check_for_dates()

def save_img(tensor,**kwargs):
    if not SAVE_IMGS:
        return
    os.makedirs(save_path, exist_ok=True)
    for i,v in enumerate(mesh.sfc_vars):
        img = tensor[0,:,:,i]
        p = '{save_path}/var={v},type={type},valid_at={valid_at},_neovis.png'.format(save_path=save_path,v=v,**kwargs)
        print(f"saving {p}")
        plt.imsave(p, img)


for snum in range(NUM_DATES):
    sample_i = len(data)//NUM_DATES*snum
    s = data[sample_i]
    t0 = s[0][1].item()
    forecast_at = get_date_str(t0)
    if forecast_at <= "2020070100":
        print("skipping", forecast_at)
        continue
    for it,dt in enumerate(timesteps):
        valid_at = get_date_str(t0+dt*3600)
        with torch.no_grad():
            x = s[0]
            x = [x[0].to(device).unsqueeze(0), torch.Tensor([x[1]]).to(device)]
            assert get_date_str(s[it+1][1]) == valid_at, f"{get_date_str(s[1][1])} {valid_at}"
            ref = s[it+1][0][:,:,-mesh.n_sfc_vars:].cpu().detach().unsqueeze(0)
            with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
                gens = model.generate(x,dt,steps=NUM_STEPS,num=NUM_GEN,plot_lines=False)

            eval_path = modelevalpath(model,model.name)
            for gi,gen in enumerate(gens):
                gen = unnorm(gen,mesh_sfc)
                save_instance(gen,f'{eval_path}/outputs/gen_{gi:02d}{get_date_str(t0)}+{dt}',mesh_sfc,model.name)

            save_img(ref,valid_at=valid_at,type='0era5')

            genmean = torch.mean(torch.stack(gens),dim=0)        
            save_img(genmean,valid_at=valid_at,type='1genmean')

            genmeanu = unnorm(genmean,mesh_sfc)
            save_instance(genmeanu,f'{eval_path}/outputs/mean_{get_date_str(t0)}+{dt}',mesh_sfc,model.name)

            ar = model.predict_ar(x,[dt]).cpu().detach()
            save_img(ar,valid_at=valid_at,type='2ar')

            if SAVE_PLOTS:
                for v in mesh.sfc_vars:
                    plt.figure(figsize=(10,5))
                    def pltline(y,label):
                        l = y[0,180,:,mesh.sfc_vars.index(v)]
                        plt.plot(np.linspace(0,360,len(l)),l,label=label)
                    pltline(ref,'era5')
                    pltline(genmean,'genmean')
                    pltline(ar,'ar')
                    #for j,gen in enumerate(gens):
                    #    pltline(gen,f'gen{j:02d}')
                    plt.grid()
                    plt.title(f'{v} vs longitude at lat=45N, valid_at={valid_at}')
                    plt.xlabel('longitude')
                    plt.legend()
                    plt.tight_layout()
                    p = f'{save_path}/var={v},type=00plot,valid_at={valid_at},_neovis.png'
                    plt.savefig(p)
                    plt.close()

            for j,gen in enumerate(gens):
                save_img(gen,valid_at=valid_at,type=f'gen{j:02d}')
        
n = Neovis(save_path)
n.make()

