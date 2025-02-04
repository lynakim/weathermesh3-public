import torch
import pickle
from utils import *
from data import NeoDatasetConfig
from dataloader import NeoDataConfig
from model_latlon_3d import *


EVALUATION_PATH = '/huge/deep/evaluation/'

def package(model,load_path,name):
    checkpoint = torch.load(load_path,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    params = sum(p.numel() for p in model.parameters())
    fname = f'{name}_{int(params/1e6)}M'
    print("Number of params: %.2fM" % (params/1e6))
    os.makedirs(f'{EVALUATION_PATH}/{fname}',exist_ok=True)
    save_path = f'{EVALUATION_PATH}/{fname}/model.pickle'
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print("Saved to", save_path)

def package_Dec3():
    load_path = '/fast/windborne/deep/ignored/runs/run_Dec1-finetune_20231127-173648/model_epoch45_iter512020_loss0.021.pt'
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoDataConfig(  dataset_config=dsc,
                    timesteps=[24], max_ram_manual=int(10e9),
                    worker_complain = True,
                    requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                    )
    model = ForecastStepSwin3D(ForecastStepConfig(data.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=24, lat_compress=False, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False
    package(model,load_path,'Dec3')

def package_Pantene():
    load_path = '/fast/windborne/deep/ignored/runs/run_Dec3-pantene_20231203-143553/model_epoch0_iter59988_loss0.099.pt'
    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoDataConfig(  dataset_config=dsc,
                    timesteps=[24], max_ram_manual=int(10e9),
                    worker_complain = True,
                    requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                    )
    model = ForecastStepSwin3D(ForecastStepConfig(data.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=24, lat_compress=False, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False
    checkpoint = torch.load(load_path,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    package(model,'Pantene')

def package_Neoreal():
    load_path = '/fast/windborne/deep/ignored/runs/run_Dec4-neoreal-neoreprise_20231205-130457/model_epoch25_iter331176_loss0.089.pt'

    ## this is copied from when you start the run

    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[24], max_ram_manual=int(10e9),
                                           worker_complain = False,
                                           requested_dates = get_dates((D(2000, 1, 23),D(2017, 12,30))),
                                           only_at_z=[0,12]
                                           ))
    print("uhhh data.config is", data.config)
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=28, lat_compress=False, timesteps=[24]))
    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'Neoreal')


def package_Tardis():
    load_path = '/fast/ignored/runs/run_Dec4-tardis-reprise_20231205-234837/model_epoch2_iter257592_loss0.090.pt'

    ## this is copied from when you start the run

    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))

    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'Tardis')


def package_TardisL2():
    load_path = '/fast/ignored/runs/run_Dec4-tardis-l2ft_20231205-234837/model_epoch15_iter199188_loss0.012.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-l2ft_20231205-234837/model_epoch16_iter215988_loss0.014.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-l2ft_20231205-234837/model_epoch19_iter249588_loss0.014.pt'
    #load_path = '/fast/ignored/runs/run_Dec4-tardis-l2ft_20231205-234837/model_epoch19_iter254388_loss0.014.pt'
    #load_path = '/fast/ignored/runs/run_Dec4-tardis-l2ft_20231205-234837/model_epoch19_iter256788_loss0.013.pt'

    ## this is copied from when you start the run

    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))

    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'TardisL2FT')


def package_TardisNeoL2():
    load_path = '/fast/ignored/runs/run_Dec4-tardis-neol2ft-reprise_20231205-234837/model_epoch2_iter74388_loss0.014.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-neol2ft-reprise_20231205-234837/model_epoch2_iter67188_loss0.016.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-slowl2ft_20231205-234837/model_epoch3_iter79188_loss0.015.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-slowl2ft_20231205-234837/model_epoch8_iter213588_loss0.015.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-slowl2ft_20231205-234837/model_epoch11_iter290388_loss0.014.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-slowl2ft_20231205-234837/model_epoch12_iter331188_loss0.016.pt'

    ## this is copied from when you start the run

    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))

    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'TardisNeoL2FT')


def package_EnsSmall():
    load_path = '/fast/ignored/runs/run_Dec14-ens_20231214-183221/model_epoch2_iter254392_loss0.096.pt'

    ## this is copied from when you start the run

    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0, levels=levels_ecm2)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'EnsSmall')

def package_SmolMenace():
    load_path = '/fast/ignored/runs/run_Dec16-smolmenace-reprise_20231217-110222/model_epoch4_iter355188_loss0.093.pt'
    load_path = '/fast/ignored/runs/run_Dec16-smolmenace-ft_20231217-110222/model_epoch5_iter155988_loss0.015.pt'
    load_path = '/fast/ignored/runs/run_Dec16-smolmenace-ft_20231217-110222/model_epoch9_iter261588_loss0.016.pt'
    load_path = '/fast/ignored/runs/run_Dec16-smolmenace-ft_20231217-110222/model_epoch11_iter290388_loss0.014.pt'
    load_path = '/fast/ignored/runs/run_Dec16-smolmenace-ft_20231217-110222/model_epoch11_iter307188_loss0.014.pt'

    ## this is copied from when you start the run

    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0, levels=levels_ecm2)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model  = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'SmolMenace')


def package_SingleDec():
    load_path = '/fast/ignored/runs/run_Dec18-singledec-hr_20231218-222644/model_epoch3_iter311992_loss0.074.pt'
    load_path = '/fast/ignored/runs/run_Dec18-singledec-hr_20231218-222644/model_epoch4_iter358392_loss0.080.pt'

    ## this is copied from when you start the run

    dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)#, levels=levels_ecm2)
    data = NeoWeatherDataset(NeoDataConfig(dataset_config=dsc,
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[24], dims_per_head=32, processor_dt=12))
    model.output_deltas = False
    model.do_sub = False

    package(model,load_path,'SingleDec')


def package_Tiny():
    load_path = '/fast/ignored/runs/run_Dec19-tiny_20231220-014744/model_epoch2_iter201594_loss0.112.pt'
    load_path = '/fast/ignored/runs/run_Dec19-tiny_20231220-014744/model_epoch2_iter225594_loss0.102.pt'
    load_path = '/fast/ignored/runs/run_Dec19-tiny-ft_20231220-014744/model_epoch0_iter22794_loss0.026.pt'
    load_path = '/fast/ignored/runs/run_Dec19-tiny-ft_20231220-014744/model_epoch4_iter122394_loss0.027.pt'
    load_path = '/fast/ignored/runs/run_Dec19-tiny-ft_20231220-014744/model_epoch9_iter261594_loss0.025.pt'

    ## this is copied from when you start the run

    mesh= meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, levels=levels_tiny, source='era5-13')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    #model  = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(2,8,8), window_size=(2,6,12), checkpoint_every=1, hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32, surface_sandwich_bro=True, use_matepoint=False))
    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'Tiny2')

def package_Quadripede():
    load_path = '/fast/ignored/runs/run_Dec12-quadripede_20231212-150101/model_epoch3_iter285588_loss0.083.pt'

    ## this is copied from when you start the run

    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))

    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=2, dec_swin_depth=2, proc_swin_depth=8, lat_compress=False, timesteps=[6,24], dims_per_head=32))
    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'Quadripede')



def package_NeoEnc():
    load_path = '/fast/ignored/runs/run_Dec22-neoenc_20231222-170550/model_epoch2_iter215996_loss0.097.pt'

    ## this is copied from when you start the run

    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12,
                                                  lat_compress=False, timesteps=[24], dims_per_head=32, processor_dt=12,use_matepoint=False))
    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'NeoEnc')



def package_TinyOper():
    load_path = '/fast/ignored/runs/run_Dec19-tiny_20231220-014744/model_epoch2_iter201594_loss0.112.pt'
    load_path = '/fast/ignored/runs/run_Dec19-tiny_20231220-014744/model_epoch2_iter225594_loss0.102.pt'
    load_path = '/fast/ignored/runs/run_Dec19-tiny-ft_20231220-014744/model_epoch0_iter22794_loss0.026.pt'
    load_path = '/fast/ignored/runs/run_Dec19-tiny-ft_20231220-014744/model_epoch4_iter122394_loss0.027.pt'
    load_path = '/fast/ignored/runs/run_Dec19-tiny-ft_20231220-014744/model_epoch9_iter261594_loss0.025.pt'

    ## this is copied from when you start the run

    dsc = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, levels=levels_tiny, source='era5-13')
    dsc2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='hres-13', levels=levels_tiny)
    data = NeoWeatherDataset(NeoDataConfig(inputs=[dsc2], outputs=[dsc],
                                           timesteps=[12,24], max_ram_manual=int(5e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=512, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    #model  = ForecastStepSwin3D(ForecastStepConfig(data.config.mesh, patch_size=(2,8,8), checkpoint_every=1, hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(2,8,8), window_size=(2,6,12), checkpoint_every=1, hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32, surface_sandwich_bro=True, use_matepoint=False))
    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'TinyOper2')


def neopackage_TardisNeoL2():
    load_path = '/fast/ignored/runs/run_Dec4-tardis-neol2ft-reprise_20231205-234837/model_epoch2_iter74388_loss0.014.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-neol2ft-reprise_20231205-234837/model_epoch2_iter67188_loss0.016.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-slowl2ft_20231205-234837/model_epoch3_iter79188_loss0.015.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-slowl2ft_20231205-234837/model_epoch8_iter213588_loss0.015.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-slowl2ft_20231205-234837/model_epoch11_iter290388_loss0.014.pt'
    load_path = '/fast/ignored/runs/run_Dec4-tardis-slowl2ft_20231205-234837/model_epoch12_iter331188_loss0.016.pt'

    ## this is copied from when you start the run

    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))

    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'neoTardisNeoL2FT')

def package_WarioJAW():
    load_path = '/fast/ignored/runs_adapt/run_Dec27-Wario-JohnAlwaysWins_20231227-143456/model_epoch16_iter50994_loss0.170.pt'

    ## this is copied from when you start the run

    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='gfs-28')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1],
                                           outputs = [mesh3],
                                           timesteps=[0],
                                           requested_dates = get_dates((D(2021, 3, 20),D(2023, 4, 30))),
                                           ))
    import model_unet
    model_conf = ForecastStepConfig(data.config.inputs, 
                outputs = data.config.outputs,
                timesteps=[0], 
                output_deltas = True,
                use_matepoint = False,
                adapter_dim_seq = [64,128,256],
                adapter_H1 = 16,
                activation = nn.GELU(),
                adapter_use_input_bias = False,
                )

    model = model_unet.ForecastStepAdapterConv(model_conf)
        
    ### TODO: Fix inputing GFS Biases, Activation function = GeLU
    package(model,load_path,'WarioJAW')

def package_neoquadripede():
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch3_iter295188_loss0.083.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch3_iter355188_loss0.076.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch4_iter431988_loss0.094.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch5_iter506388_loss0.077.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch5_iter544788_loss0.072.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch5_iter547188_loss0.088.pt'

    ## this is copied from when you start the run

    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[6,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], dims_per_head=32, processor_dt=6, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=False))

    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'neoquadripede')

def package_neoquadripede2():
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch3_iter295188_loss0.083.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch3_iter355188_loss0.076.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch4_iter431988_loss0.094.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch5_iter506388_loss0.077.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch5_iter544788_loss0.072.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch5_iter547188_loss0.088.pt'

    ## this is copied from when you start the run

    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[6,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], dims_per_head=32, processor_dt=6, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=False))

    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'neoquadripede_neohegel')



def package_neoquadripede48():
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede-beta72_20231229-015824/model_epoch4_iter86388_loss0.020.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede-beta72_20231229-015824/model_epoch9_iter158388_loss0.015.pt'
    load_path = '/fast/ignored/runs/run_Dec28-neoquadripede-beta72_20231229-015824/model_epoch11_iter194388_loss0.015.pt'

    ## this is copied from when you start the run

    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[6,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], dims_per_head=32, processor_dt=6, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=False))

    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'neoquadripede48')


def package_singlemenace():
    load_path = '/fast/ignored/runs/run_Dec29-singlemenace-reprise_20240103-204449/model_epoch2_iter243990_loss0.074.pt'
    load_path = '/fast/ignored/runs/run_Dec29-singlemenace-reprise_20240103-204449/model_epoch2_iter261990_loss0.075.pt'
    load_path = '/fast/ignored/runs/run_Dec29-singlemenace-reprise_20240103-204449/model_epoch2_iter275990_loss0.073.pt'
    load_path = '/fast/ignored/runs/run_Dec29-singlemenace-reprise_20240103-204449/model_epoch2_iter291990_loss0.073.pt'

    ## this is copied from when you start the run

    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[6,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=1024, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[24], dims_per_head=32, processor_dt=6, use_matepoint=False, output_deltas=False))
    model.output_deltas = False
    model.do_sub = False

    package(model,load_path,'singlemenace')
    package(model,load_path,'singlemenace_120x24')

def package_casio():
    load_path = '/fast/ignored/runs/run_Jan10-casio_20240110-164835/model_epoch1_iter155992_loss0.066.pt'

    ## this is copied from when you start the run

    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    timesteps = [1,3,6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], dims_per_head=32, processor_dt=6, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=False))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=False, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64))
    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'casio')


def package_neocasio():
    load_path = '/fast/ignored/runs/run_Jan16-neocasio_20240116-183046/model_epoch2_iter226392_loss0.057.pt'

    ## this is copied from when you start the run

    #dsc = NeoDatasetConfig(WEATHERBENCH=1, CLOUD=0)
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28',extra_sfc_vars=extra, output_only_vars=extra)
    timesteps = [1,3,6]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=timesteps, max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, lat_compress=False, timesteps=[12,24], dims_per_head=32))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], dims_per_head=32, processor_dt=6, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=False))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=False, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=False, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64, neorad=True))
    model.output_deltas = True
    model.do_sub = False

    package(model,load_path,'neocasio')



if __name__ == "__main__":
    #package_Neoreal()
    #package_Tardis()
    #package_TardisL2()
    #neopackage_TardisNeoL2()
    #package_EnsSmall()
    #package_SmolMenace()
    #package_SingleDec()
    #package_Tiny()
    #package_TinyOper()
    #package_Quadripede()
    #package_NeoEnc()
    #package_WarioJAW() 
    #package_neoquadripede48()
    package_neoquadripede2()
    #package_casio()
    #package_neocasio()
    #package_singlemenace()
