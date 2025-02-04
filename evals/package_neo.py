import torch
import pickle
import sys
sys.path.append('.')
from utils import *
from model_latlon_3d import *
import shutil
from diffusion.model import UNet
from model_latlon.diffusion import ForecastCombinedDiffusion

EVALUATION_PATH = '/huge/deep/evaluation/'
if not os.path.exists(EVALUATION_PATH):
    EVALUATION_PATH = '/fast/ignored/evaluation/'

STRICT = True

def modelfullname(model,name):
    #params = sum(p.numel() for p in model.parameters())
    #return f'{name}_{int(params/1e6)}M'
    # ^ oof, im sorry about this it turns out putting the number of params in the path was an extremely bad idea
    # all the if statement below are just backwards compatibility for the old names
    if name == 'yamaha':
        return 'yamaha_168M'
    if name == 'serp3bachelor':
        return 'serp3bachelor_168M'
    if name.startswith('rtyamahabachelor'):
        return name + '_328M'
    if name == 'yamahabachelor':
        return name + '_286M'
    elif name.startswith('yamahabachelor'):
        return name + '_245M'
    return name

def modelevalpath(model,name):
    return f'{EVALUATION_PATH}/{modelfullname(model,name)}'

def weightspath(model,load_path,name):
    if load_path == 'resave':
        resaves = [f for f in os.listdir(f'{modelevalpath(model,name)}/weights') if f.startswith('resave')]
        assert len(resaves) > 0, "No resaves found"
        resaves.sort()
        return f'{modelevalpath(model,name)}/weights/{resaves[-1]}'
    
    return f'{modelevalpath(model,name)}/weights/{os.path.basename(load_path)}'




def package_(model,load_path,name,strict=STRICT):
    print("New package, no pickles")
    print("uhh", torch)
    print("aa", load_path)
    checkpoint = torch.load(load_path,map_location='cpu')
    sd = model_state_dict_name_mapping(checkpoint['model_state_dict'],model)
    model.load_state_dict(sd,strict=strict)
    save_path = weightspath(model,load_path,name)
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    shutil.copy(load_path, save_path)
    print("Saved to", save_path)
    return model

def load_weights(model,weights,name='',strict=STRICT):
    print("Loading weights from", weights)
    checkpoint = torch.load(weights,map_location='cpu')
    if 'model_state_dict' in checkpoint:
        sd = checkpoint['model_state_dict']
    else:
        sd = checkpoint
    sd = model_state_dict_name_mapping(sd,model)
    if not strict:
        # relevant zulip convo https://chat.windbornesystems.com/#narrow/stream/201-tech-dlnwp/topic/misc/near/6768175
        print("😱😱😱 WARNING: loading weights with strict=False, your outputs may be garbage 😱😱😱") 
    else:
        sd_keys = set(sd.keys())
        model_keys = set(model.state_dict().keys())
        assert sd_keys == model_keys, f"Extra sd keys: {sd_keys - model_keys} \nExtra model keys: {model_keys - sd_keys}"
    model.load_state_dict(sd,strict=strict) 

def package(load_path=None,strict=STRICT):
    def inner(model_func):
        assert model_func.__name__.startswith('get_'), "model_func must be named get_something"
        name = model_func.__name__[4:]
        def wrapper(resave=False, no_load=False):
            model = model_func()
            model.name = name
            if load_path is not None:
                weights = weightspath(model,load_path,name)
                if no_load:
                    assert resave == False, "no_load and resave are mutually exclusive"
                    print(f"Not loading {weights} because no_load=True")
                    return model
                print("loading", weights)
                if not os.path.exists(weights):
                    return package_(model,load_path,name,strict=strict)
                load_weights(model,weights,name,strict=strict)
                # save state dict if only whole-model checkpoint exists
                state_dict_name = weights.split("/")[-1].replace("model", "state_dict")
                state_dict_path = "/".join(weights.split("/")[:-1]) + "/" + state_dict_name
                if not os.path.exists(state_dict_path):
                    print("Only whole-model checkpoint exists")
                    print(f"Saving state dict to {state_dict_path}")
                    torch.save(model.state_dict(), state_dict_path)
            if resave:
                assert sys.argv[0] == 'evals/package_neo.py', "I'm not resaving unless you called this from the right place"
                os.makedirs(f'{modelevalpath(model,name)}/weights',exist_ok=True)
                resave_path =  f'{modelevalpath(model,name)}/weights/resave_{time.strftime("%Y%m%d-%H%M%S")}.pt'
                print("Resaving to", resave_path)
                torch.save(model.state_dict(), resave_path)

            return model 
        wrapper.__name__ = model_func.__name__
        wrapper.modelname = name
        return wrapper
    return inner

@package('/huge/deep/runs/run_Dec28-operational-stackedconvplus_20241227-221550/model_epoch3_iter445188_step37099_loss0.012.pt')
def get_operational_stackedconvplus(): 
    tdates = get_dates([(D(2024, 2, 1), D(2024, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h',
             'tc-maxws', 'tc-minp'] # New TC outputs!

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            requested_dates = tdates
                                            ))

    import model_latlon.top as top
    import model_latlon.encoder as encoders
    import model_latlon.decoder as decoders
    conf = top.ForecastModelConfig(
        data.config.inputs,
        latent_size = 1440, # Highly composite number, larger latent space (was 1280)
        affine = True,
        pr_depth = [8],
        encdec_tr_depth = 3, # More Transformers! (was 2)
        tr_embedding = 'rotary',
        patch_size = (4,8,8),
        weight_eps = 0.02,
    )
    
    encoder = encoders.SimpleConvEncoder(imesh, conf)
    decoder = decoders.StackedConvPlusDecoder(omesh, conf)
    model = top.ForecastModel(
        conf,
        encoders=[encoder],
        decoders=[decoder]
    )
    
    return model

@package('/huge/deep/runs/run_Dec16-TCregionalio_20241216-132513/model_epoch0_iter15998_step7999_loss0.150.pt')
def get_TCregionalio():
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_medium)
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_medium)
    output_mesh_2 = meshes.TCRegionalLocations()
    
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvEncoder
    from model_latlon.decoder import SimpleConvPlusDecoder, RegionalTCDecoder # StackedConvPlusDecoder
    
    conf = ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth=4,
        latent_size=896,
        window_size=(3,5,7)
    )
    
    model = ForecastModel(
        input_mesh, 
        conf,
        encoders=[SimpleConvEncoder(input_mesh, conf)],
        decoders=[
            SimpleConvPlusDecoder(input_mesh, conf),
            RegionalTCDecoder(input_mesh, 
                              conf, 
                              region_radius=2, 
                              tr_depth=2,
                              tr_window_size=(3, 5, 7))
            ])
    
    return model

@package('/huge/deep/runs/run_Dec16-TCregional_20241216-131930/model_epoch0_iter15998_step7999_loss0.012.pt')
def get_TCregional():
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28))])
    extra = ['tc-maxws', 'tc-minp']
    input_mesh = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_pad=len(extra), 
        input_levels=levels_medium,
        levels=levels_medium)
    output_mesh_1 = meshes.LatLonGrid(
        source='era5-28', 
        extra_sfc_vars=extra, 
        input_levels=levels_medium,
        levels=levels_medium)
    output_mesh_2 = meshes.TCRegionalLocations()
    
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(
        inputs=[input_mesh],
        outputs=[output_mesh_1, output_mesh_2],
        timesteps=timesteps,
        requested_dates=tdates
    ))
    
    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvEncoder
    from model_latlon.decoder import SimpleConvPlusDecoder, RegionalTCDecoder # StackedConvPlusDecoder
    
    conf = ForecastModelConfig(
        data.config.inputs,
        outputs=data.config.outputs,
        encdec_tr_depth=4,
        latent_size=896,
        window_size=(3,5,7)
    )
    
    model = ForecastModel(
        input_mesh, 
        conf,
        encoders=[SimpleConvEncoder(input_mesh, conf)],
        decoders=[
            SimpleConvPlusDecoder(input_mesh, conf),
            RegionalTCDecoder(input_mesh, 
                              conf, 
                              region_radius=2, 
                              tr_depth=2, 
                              tr_window_size=(3, 5, 7))
            ])
    
    return model

@package('/huge/deep/runs/run_Jul29-bachelor_20240801-205915/model_epoch3_iter338394_step56399_loss0.051.pt')
def get_bachelor():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 1))])
    timesteps = [6,24]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                     timesteps=timesteps,
                                     requested_dates=tdates))
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs, 
            outputs=data.config.outputs, 
            sincos=True, 
            padded_lon=True, 
            Transformer=SlideLayer3D, 
            patch_size=(5,8,8), 
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            output_deltas=False, 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96, 
            neorad=True, 
            window_size=(3,5,7)))
    
    return model

@package('/huge/deep/runs/run_Nov22-TCfullforce-try2_20241122-151150/model_epoch0_iter42598_step21299_loss0.006.pt')
def get_TCfullforce():
    extra = ['tc-maxws', 'tc-minp']
    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)

    tdates = get_dates([(D(1971, 1, 1), D(2022, 12, 31), timedelta(hours=3))])
    timesteps = [6,24,72]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                     timesteps=timesteps,
                                     requested_dates=tdates,
                                     only_at_z=list(range(0, 24, 3))))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=False, 
            neorad=True, 
            window_size=(3,5,7)))
    return model

@package('/huge/deep/runs/run_Nov21-TCvariableweights-try2_20241122-150927/model_epoch0_iter50998_step25499_loss0.005.pt')
def get_TCvarweight():
    extra = ['tc-maxws', 'tc-minp']
    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=len(extra), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)

    tdates = get_dates([(D(1971, 1, 1), D(2022, 12, 31), timedelta(hours=3))])
    timesteps = [6,24,72]
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                     timesteps=timesteps,
                                     requested_dates=tdates,
                                     only_at_z=list(range(0, 24, 3))))
    
    model = ForecastStep3D(
        ForecastStepConfig(
            data.config.inputs,
            outputs=data.config.outputs,
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=False, 
            neorad=True, 
            window_size=(3,5,7)))
    return model
    
@package('/fast/ignored/runs/run_Dec28-neoquadripede-beta72_20231229-015824/model_epoch11_iter201588_loss0.015.pt')
def get_neoquadripede48():
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[6,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(
        ForecastStepConfig(data.config.inputs, 
                           outputs=data.config.outputs, 
                           patch_size=(4,8,8), 
                           hidden_dim=896, 
                           enc_swin_depth=4, 
                           dec_swin_depth=4, 
                           proc_swin_depth=8, 
                           lat_compress=False, 
                           timesteps=[6, 24], 
                           dims_per_head=32, 
                           processor_dt=6, 
                           decoder_reinput_initial=True, 
                           decoder_reinput_size=96, 
                           use_matepoint=False))
    model.output_deltas = True
    model.do_sub = False
    return model

@package('/fast/ignored/runs/run_Feb5-neoquad-ft2_20231229-015824/model_epoch3_iter108792_loss0.006.pt')
def get_shortquad():
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[6], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6], dims_per_head=32, processor_dt=6, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=False))
    model.output_deltas = True
    model.do_sub = False
    return model




@package('/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch5_iter547188_loss0.088.pt')
def get_neoquadripede():
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[6,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], dims_per_head=32, processor_dt=6, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=False))
    model.output_deltas = True
    model.do_sub = False
    return model

@package('/fast/ignored/runs/run_Dec28-neoquadripede_20231229-015824/model_epoch2_iter196788_loss0.104.pt')
def get_yungneoquadripede():
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[6,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], dims_per_head=32, processor_dt=6, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=False))
    model.output_deltas = True
    model.do_sub = False
    return model

@package('/fast/ignored/runs/run_Dec19-tiny-ft_20231220-014744/model_epoch9_iter261594_loss0.025.pt')
def get_tiny():
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, levels=levels_tiny, source='era5-13')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],
                                           timesteps=[12,24], max_ram_manual=int(10e9),
                                           worker_complain = not True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(2,8,8), window_size=(2,6,12), checkpoint_every=1, hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=14, lat_compress=False, timesteps=[12,24], dims_per_head=32, surface_sandwich_bro=True, use_matepoint=False))
    model.output_deltas = True
    return model

@package('/fast/ignored/runs/run_Jan16-neocasio_20240116-183046/model_epoch2_iter271192_loss0.051.pt')
def get_neocasio():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_model_epoch93_iter89982_loss0.066.ptvars=extra)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    timesteps = [1, 3, 6]
    model = ForecastStepSwin3D(ForecastStepConfig([input], outputs=[output], patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=False, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64, neorad=True))
    return model

@package('/fast/ignored/runs/run_Jan29-shortking-reprise_20240129-144601/model_epoch4_iter424782_loss0.067.pt')
def get_shortking():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_model_epoch93_iter89982_loss0.066.ptvars=extra)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    timesteps = [1, 3, 6]
    #model = ForecastStepSwin3D(ForecastStepConfig([input], outputs=[output], patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=False, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64, neorad=True))
    model = ForecastStepSwin3D(ForecastStepConfig([input], outputs=[output], patch_size=(4,8,8), hidden_dim=1024, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=16, lat_compress=False, timesteps=[6], train_timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=False, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64, neorad=True))
    return model


@package('/fast/ignored/runs/run_Feb8-shortking-replay1_20240129-144601/model_epoch6_iter278982_loss0.083.pt')
def get_rpshortking():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_model_epoch93_iter89982_loss0.066.ptvars=extra)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    timesteps = [1, 3, 6]
    #model = ForecastStepSwin3D(ForecastStepConfig([input], outputs=[output], patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=6, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=1, use_matepoint=False, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64, neorad=True))
    model = ForecastStepSwin3D(ForecastStepConfig([input], outputs=[output], patch_size=(4,8,8), hidden_dim=1024, enc_swin_depth=0, dec_swin_depth=0, proc_swin_depth=16, lat_compress=False, timesteps=[6], train_timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=False, output_deltas=True, decoder_reinput_initial=False, decoder_reinput_size=64, neorad=True))
    return model

#@package('/fast/ignored/runs/run_Feb16-widepony_20240217-022407/model_epoch1_iter158382_loss0.111.pt')
#@package('/fast/ignored/runs/run_Feb16-widepony_20240217-022407/model_epoch1_iter188982_loss0.119.pt')
#@package('/fast/ignored/runs/run_Feb16-widepony_20240217-022407/model_epoch2_iter241182_loss0.130.pt')
#@package('/fast/ignored/runs/run_Feb16-widepony-reprise_20240217-022407/model_epoch0_iter44982_loss0.062.pt') <-- bad?
#@package('/fast/ignored/runs/run_Feb16-widepony-reprise_20240217-022407/model_epoch0_iter97182_loss0.058.pt')
#@package('/fast/ignored/runs/run_Feb16-widepony-reprise_20240217-022407/model_epoch1_iter129582_loss0.061.pt')
#@package('/fast/ignored/runs/run_Feb16-widepony-reprise_20240217-022407/model_epoch1_iter172782_loss0.058.pt')
# @package('/fast/ignored/runs/run_Feb16-widepony-reprise_20240217-022407/model_epoch2_iter219582_loss0.057.pt')
@package('/fast/ignored/runs/run_Feb16-widepony-reprise_20240217-022407/model_epoch6_iter597582_loss0.060.pt')
def get_widepony():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_model_epoch93_iter89982_loss0.066.ptvars=extra)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    timesteps = [24, 72]
    model = ForecastStepSwin3D(ForecastStepConfig([input], outputs=[output], patch_size=(4,8,8), hidden_dim=928, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=10, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=False, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(4,3,6)))
    return model

#@package('/fast/ignored/runs/run_Feb18-shallowpony_20240218-184013/model_epoch1_iter133592_loss0.105.pt')
#@package('/fast/ignored/runs/run_Feb18-shallowpony_20240218-184013/model_epoch3_iter298392_loss0.097.pt')
#@package('/fast/ignored/runs/run_Feb18-shallowpony_20240218-184013/model_epoch3_iter327992_loss0.100.pt')
#@package('/fast/ignored/runs/run_Feb18-shallowpony_20240218-184013/model_epoch3_iter372792_loss0.095.pt')
#@package('/fast/ignored/runs/run_Feb18-shallowpony_20240218-184013/model_epoch4_iter421592_loss0.091.pt')
@package('/fast/ignored/runs/run_Feb18-shallowpony_20240218-184013/model_epoch4_iter479992_loss0.088.pt')
def get_shallowpony():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_model_epoch93_iter89982_loss0.066.ptvars=extra)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24, 72]
    timesteps = [24]
    #model = ForecastStepSwin3D(ForecastStepConfig([input], outputs=[output], patch_size=(4,8,8), hidden_dim=928, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=10, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=False, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(4,3,6)))
    model = ForecastStepSwin3D(ForecastStepConfig([input], outputs=[output], patch_size=(5,8,8), hidden_dim=1504, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=4, lat_compress=False, timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=False, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=128, neorad=True, window_size=(2,3,6)))
    return model



@package(None)
def get_neocasioquad():
    model1 = get_neoquadripede()
    model2 = get_neocasio()
    model = ForecastStepCombo([model2, model1])
    return model

#@package("/fast/ignored/runs_adapt/run_Jan23-schelling_20240123-193116/model_epoch26_iter25196_loss0.068.pt")
#@package("/fast/ignored/runs_adapt/run_Jan31-legeh_20240131-173243/model_epoch93_iter89982_loss0.066.pt")
#@package("/fast/ignored/runs_adapt/run_Feb1-legeh-911_20240201-133247/model_epoch16_iter16182_loss0.060.pt")
#@package("/fast/ignored/runs_adapt/run_Feb1-legeh-911_20240201-133247/model_epoch18_iter17982_loss0.062.pt")
#@package("/fast/ignored/runs_adapt/run_Feb1-legeh-911_20240201-133247/model_epoch44_iter43182_loss0.058.pt")
@package("/fast/ignored/runs_adapt/run_Feb1-legeh-911-reprise_20240201-133247/model_epoch56_iter54392_loss0.057.pt")
def get_neohegel():
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25', input_levels=levels_gfs, levels=levels_medium)
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    import model_unet
    model_conf = ForecastStepConfig([mesh1, mesh2],
                outputs = [mesh3],
                patch_size=(4,8,8), 
                hidden_dim=768, 
                enc_swin_depth=0,
                dec_swin_depth=0, 
                proc_swin_depth=0, 
                adapter_swin_depth=8,
                timesteps=[0], 
                output_deltas = True,
                adapter_use_input_bias=True,
                use_matepoint = False,
                processor_dt = -1,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True),
                )

    adapter_model = model_unet.ForecastStepAdapterConv(model_conf)
    return adapter_model

@package("/fast/ignored/runs_adapt/run_Feb9-schopenhowitzer-reprise_20240209-103746/model_epoch19_iter19192_loss0.068.pt")
def get_schopenhowitzer():
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25', input_levels=levels_gfs, levels=levels_medium)
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    import model_unet
    model_conf = ForecastStepConfig([mesh1,mesh2],
                outputs = [mesh3],
                patch_size=(4,8,8),
                hidden_dim=768,
                enc_swin_depth=0,
                dec_swin_depth=0,
                proc_swin_depth=0,
                adapter_swin_depth=0,
                timesteps=[0],
                adapter_use_input_bias=True,
                output_deltas = True,
                use_matepoint = False,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True)
                )
    adapter_model = model_unet.ForecastStepAdapterConv(model_conf)
    return adapter_model

@package("/fast/ignored/runs_adapt/run_Feb1-legeh-911-ft_20240201-133247/model_epoch72_iter91192_loss0.077.pt")
def get_legeh2():
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25', input_levels=levels_gfs, levels=levels_medium)
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    import model_unet
    model_conf = ForecastStepConfig([mesh1,mesh2],
                outputs = [mesh3],
                patch_size=(4,8,8),
                hidden_dim=768,
                enc_swin_depth=0,
                dec_swin_depth=0,
                proc_swin_depth=0,
                adapter_swin_depth=0,
                timesteps=[0],
                adapter_use_input_bias=True,
                output_deltas = True,
                use_matepoint = False,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True)
                )
    adapter_model = model_unet.ForecastStepAdapterConv(model_conf)
    return adapter_model


@package("/fast/ignored/runs_adapt/run_Feb1-legeh-911-reprise_20240201-133247/model_epoch56_iter54392_loss0.057.pt")
def get_legeh():
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25', input_levels=levels_gfs, levels=levels_medium)
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    import model_unet
    model_conf = ForecastStepConfig([mesh1,mesh2],
                outputs = [mesh3],
                patch_size=(4,8,8),
                hidden_dim=768,
                enc_swin_depth=0,
                dec_swin_depth=0,
                proc_swin_depth=0,
                adapter_swin_depth=0,
                timesteps=[0],
                adapter_use_input_bias=True,
                output_deltas = True,
                use_matepoint = False,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True)
                )
    adapter_model = model_unet.ForecastStepAdapterConv(model_conf)
    return adapter_model

@package(None)
def get_hegelcasioquad():
    model1 = get_neoquadripede()
    model2 = get_neocasio()
    adapter = get_neohegel()
    model = ForecastStepCombo([model2,model1],adapter=adapter)
    return model


@package(None)
def get_hegelcasiopony():
    model1 = get_shallowpony()
    model2 = get_neocasio()
    adapter = get_neohegel()
    model = ForecastStepCombo([model1, model2],adapter=adapter)
    return model


@package(None)
def get_hegelcasio():
    model = get_neocasio()
    adapter = get_neohegel()
    model = ForecastStepCombo([model],adapter=adapter)
    return model

@package(None)
def get_hegelquad():
    model = get_neoquadripede()
    adapter = get_neohegel()
    model = ForecastStepCombo([model],adapter=adapter)
    return model

@package(None)
def get_yunghegelquad():
    model = get_yungneoquadripede()
    adapter = get_legeh()
    model = ForecastStepCombo([model],adapter=adapter)
    return model

@package(None)
def get_hegel2quad():
    model = get_neoquadripede()
    adapter = get_legeh2()
    model = ForecastStepCombo([model],adapter=adapter)
    return model


@package(None)
def get_hegelhandfist():
    model = get_handfist()
    adapter = get_neohegel()
    model = ForecastStepCombo([model],adapter=adapter)
    return model


@package(None)
def get_hegelquad48():
    model = get_neoquadripede48()
    adapter = get_neohegel()
    model = ForecastStepCombo([model],adapter=adapter)
    return model

@package(None)
def get_shorthegel():
    model = get_shortking()
    adapter = get_legeh()
    model = ForecastStepCombo([model],adapter=adapter)
    return model

@package(None)
def get_shorthowitzer():
    model = get_shortking()
    adapter = get_schopenhowitzer()
    model = ForecastStepCombo([model],adapter=adapter)
    return model


@package(None)
def get_rpshorthegel():
    model = get_rpshortking()
    adapter = get_legeh()
    model = ForecastStepCombo([model],adapter=adapter)
    return model


#@package('/fast/ignored/runs/run_Jan16-handfist-neo2reprise_20240116-213359/model_epoch5_iter511182_loss0.092.pt')
#@package('/fast/ignored/runs/run_Jan16-handfist-neo2reprise_20240116-213359/model_epoch5_iter557982_loss0.091.pt')
@package('/fast/ignored/runs/run_Jan16-handfist-neo2reprise_20240116-213359/model_epoch6_iter601182_loss0.089.pt')
def get_handfist():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    timesteps = [12,24]
    #mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0, source='era5-28')
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    data = NeoWeatherDataset(NeoDataConfig(inputs=[input], outputs=[output],
                                           timesteps=[6,24], max_ram_manual=int(10e9),
                                           worker_complain = True,
                                           requested_dates = get_dates((D(1979, 3, 28),D(2017, 12,30))),
                                           ))
    #model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[6, 24], dims_per_head=32, processor_dt=6, decoder_reinput_initial=True, decoder_reinput_size=96, use_matepoint=False))
    model = ForecastStepSwin3D(ForecastStepConfig(data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), hidden_dim=832, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=[12,24], train_timesteps=timesteps, dims_per_head=32, processor_dt=6, use_matepoint=False, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=64, neorad=True))
    model.output_deltas = True
    model.do_sub = False
    return model

#@package('/fast/ignored/runs/run_Jul19-wrap-rpb-epb-fp32_20240720-020553/model_epoch2_iter267396_step34599_loss0.099.pt')
#@package('/fast/ignored/runs/run_Jul19-wrap-rpb-epb-fp32_20240720-020553/model_epoch2_iter269796_step34799_loss0.097.pt')
#@package('/fast/ignored/runs/run_Jul19-wrap-rpb-epb-fp32_20240720-020553/model_epoch3_iter364596_step42699_loss0.096.pt')
@package('/fast/ignored/runs/run_Jul19-wrap-rpb-epb-fp32_20240720-020553/model_epoch4_iter416196_step46999_loss0.097.pt')
def get_wrappy():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24, 72]
    timesteps = [24]
    model = ForecastStepSwin3D(ForecastStepConfig(inputs=[input], outputs=[output], Transformer=SlideLayer3D, checkpointfn=None,patch_size=(5,8,8), checkpoint_every=np.nan,
                                              hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32,
                                              processor_dt=6, use_matepoint=False, output_deltas=True, decoder_reinput_initial=True, decoder_reinput_size=128, neorad=True, window_size=(3,5,5)))
    return model

@package('/huge/deep/runs/run_Jul29-bachelor_20240801-205915/model_epoch3_iter339594_step56599_loss0.057.pt')
def get_og_bachelor():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    model_conf = ForecastStepConfig([imesh], 
                    outputs=[omesh], 
                    sincos=True, 
                    padded_lon=True, 
                    Transformer=SlideLayer3D, 
                    checkpointfn=matepoint.checkpoint,
                    patch_size=(5,8,8),
                    hidden_dim=896, 
                    enc_swin_depth=4, 
                    dec_swin_depth=4, 
                    proc_swin_depth=8,
                    timesteps=[6,24], 
                    dims_per_head=32,
                    processor_dt=6,
                    output_deltas=False, 
                    decoder_reinput_initial=True, 
                    decoder_reinput_size=96, 
                    neorad=True, 
                    window_size=(3,5,7)
                )
    model = ForecastStep3D(model_conf)
    return model

#@package('/huge/deep/runs/run_Aug19-master48ft_20240819/model_epoch4_iter304788_step25399_loss0.043.pt')
#@package('/huge/deep/runs/run_Aug14-master24fast_20240814-181622/model_epoch3_iter259788_step23299_loss0.056.pt')
#@package('/huge/deep/runs/run_Aug14-master24fast_20240814-181622/model_epoch4_iter289788_step28299_loss0.059.pt')
@package('/huge/deep/runs/run_Aug14-master24fast_20240814-181622/model_epoch4_iter307188_step31199_loss0.048.pt')
def get_master():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    # input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    # #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    # output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    # #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    # output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24, 72]
    timesteps = [24,48,72,96,120]
    timesteps = [24]
    model = ForecastStep3D(
        ForecastStepConfig(
            inputs=[input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=1280,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=6,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
        window_size=(3,5,7)))


    #model = ForecastStepSwin3D(ForecastStepConfig(inputs=[input], outputs=[output], sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpoint_every=np.nan, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8),
    #                                          hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32,
    #                                          processor_dt=6, use_matepoint=False, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,7)))
    return model

#@package('/huge/deep/runs/run_Aug25-serpentbachelor3_20240825-192152/model_epoch0_iter42594_step7099_loss0.040.pt')
#@package('/huge/deep/runs/run_Aug25-serpentbachelor3_20240825-192152/model_epoch0_iter56994_step9499_loss0.044.pt')
#@package('/huge/deep/runs/run_Aug25-serpentbachelor3_20240825-192152/model_epoch1_iter78594_step13099_loss0.040.pt')
#@package('/huge/deep/runs/run_Aug25-serpentbachelor3_20240825-192152/model_epoch1_iter93594_step15599_loss0.039.pt')
@package('resave')
def get_serp3bachelor():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    input = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=6,
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
            checkpointfn=None,
        window_size=(3,5,7)))
    return model

#@package('/huge/deep/runs/run_Aug28-neomaster48_20240828-012345/model_epoch0_iter41382_step2299_loss0.039.pt')
#@package('/huge/deep/runs/run_Aug28-neomaster48_20240828-012345/model_epoch1_iter89982_step4999_loss0.046.pt')
#@package('/huge/deep/runs/run_Aug30-neomaster48_20240830-012345/model_epoch0_iter34788_step2899_loss0.049.pt')
#@package('/huge/deep/runs/run_Aug30-neomaster48_20240830-012345/model_epoch1_iter121188_step10099_loss0.053.pt')
@package('/huge/deep/runs/run_Aug30-neomaster48_20240830-012345/model_epoch1_iter164388_step13699_loss0.048.pt')
def get_longmaster():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=[], extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra, input_levels=levels_medium, levels=levels_joank)

    timesteps = [24, 72]
    timesteps = [24,48,72,96,120]
    timesteps = [24]
    timesteps = [24, 48, 72, 96, 120, 144, 168, 192, 216]
    model = ForecastStep3D(
        ForecastStepConfig(
            inputs=[input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=1280,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=6,
            timesteps=timesteps,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
        window_size=(3,5,7)))


    #model = ForecastStepSwin3D(ForecastStepConfig(inputs=[input], outputs=[output], sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpoint_every=np.nan, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8),
    #                                          hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32,
    #                                          processor_dt=6, use_matepoint=False, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,7)))
    return model

#@package('/huge/deep/runs/run_Aug30-highschooler_20240830-222057/model_epoch2_iter257994_step42999_loss0.068.pt')
@package('/huge/deep/runs/run_Aug30-highschooler_20240830-222057/model_epoch3_iter335994_step55999_loss0.059.pt')
def get_highschooler():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24, 72]
    timesteps = [24,48,72,96,120]
    timesteps = [24]
    #timesteps = [24, 48, 72, 96, 120, 144, 168, 192, 216]
    model = ForecastStep3D(
        ForecastStepConfig(
            inputs=[input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,16,16),
            hidden_dim=1024,
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            timesteps=timesteps,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
        window_size=(3,5,7)))

    #model = ForecastStepSwin3D(ForecastStepConfig(inputs=[input], outputs=[output], sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpoint_every=np.nan, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8),
    #                                          hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32,
    #                                          processor_dt=6, use_matepoint=False, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,7)))
    return model



#@package("/huge/deep/runs/run_Aug28-bachelor-gfshres-ft-serp3-2k_20240903-095614/model_epoch10_iter10497_step3499_loss0.065.pt")
@package("/huge/deep/runs/run_Sep11-bachelor-gfshres-ft-serp3-2k_20240912-021709/model_epoch7_iter7197_step2399_loss0.050.pt")
#@package("resave")
def get_rtbachelor():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25',extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny,levels=levels_joank)
    omesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    #timesteps = [6, 24]
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2], 
            outputs=[omesh], 
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=None,
            patch_size=(5,8,8), 
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=6, 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    return model

#@package('/huge/deep/runs/model_epoch0_iter47992_step5999_loss0.042.pt')
#@package('/huge/deep/runs/model_epoch0_iter48792_step6099_loss0.047.pt')
#@package('/huge/deep/runs/model_epoch0_iter69592_step8699_loss0.048.pt')
#@package('/huge/deep/runs/model_epoch0_iter70392_step8799_loss0.043.pt')
@package('/huge/deep/runs/model_epoch0_iter71992_step8999_loss0.046.pt')
def get_ultralongmaster():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=[], extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            inputs=[input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=1280,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=6,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
        window_size=(3,5,7)))


    #model = ForecastStepSwin3D(ForecastStepConfig(inputs=[input], outputs=[output], sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpoint_every=np.nan, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8),
    #                                          hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32,
    #                                          processor_dt=6, use_matepoint=False, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,7)))
    return model

@package('/huge/deep/runs/run_Sep10-relhum_20240910-170718/model_epoch2_iter302495_step60499_loss0.084.pt')
def get_relhum():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=[], extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            inputs=[input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,16,16),
            hidden_dim=1024,
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
        window_size=(3,5,7)))


    #model = ForecastStepSwin3D(ForecastStepConfig(inputs=[input], outputs=[output], sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpoint_every=np.nan, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8),
    #                                          hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32,
    #                                          processor_dt=6, use_matepoint=False, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,7)))
    return model


#@package('/huge/deep/runs/run_Sep7-doctorate_20240907-113206/model_epoch2_iter254388_step21199_loss0.035.pt')
#@package('/huge/deep/runs/run_Sep7-doctorate_20240907-113206/model_epoch2_iter255588_step21299_loss0.026.pt')
@package('/huge/deep/runs/run_Sep7-doctorate_20240907-113206/model_epoch2_iter256788_step21399_loss0.029.pt')
def get_doctorate():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only

    input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28', extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28', extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_joank)

    model = ForecastStep3D(
        ForecastStepConfig(
            inputs=[input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=1536, #1408, 
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
        window_size=(3,5,7)))


    #model = ForecastStepSwin3D(ForecastStepConfig(inputs=[input], outputs=[output], sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpoint_every=np.nan, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8),
    #                                          hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32,
    #                                          processor_dt=6, use_matepoint=False, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,7)))
    return model



@package("/huge/deep/runs/run_Sep4-s2s-stupidyolo_20240904-112227/model_epoch0_iter91194_step15199_loss0.137.pt")
def get_stupidyolo():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    timesteps = [24,24*6,24*12,24*24]
    model = ForecastStep3D(
        ForecastStepConfig(
            inputs=[imesh], 
            outputs=[omesh], 
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=matepoint.checkpoint,
            patch_size=(5,16,16), 
            hidden_dim=1024, 
            enc_swin_depth=8, 
            dec_swin_depth=8, 
            proc_swin_depth=8, 
            timesteps=timesteps, 
            dims_per_head=32, 
            processor_dt=24, 
            neorad=True,
        window_size=(3,5,7)))
    return model
    

#@package("/huge/deep/runs/run_Sep6-yamaha-bigsad_20240907-212316/model_epoch2_iter187794_step31299_loss0.012.pt")
@package("resave")
def get_yamaha():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    input = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [input], 
            outputs=[output], 
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            patch_size=(5,8,8), 
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8, 
            dims_per_head=32, 
            processor_dt=1,
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True,
        window_size=(3,5,7)))
    return model
    
#@package(load_path='resave')
#@package()
#@package('/huge/deep/runs/run_Sep12-YB-fullfinetune_20240912-234359/model_epoch0_iter3996_step999_loss0.076.pt')
#@package('/huge/deep/runs/run_Sep14-YB-ft1hr24-rand_20240915-115807/model_epoch0_iter10396_step2599_loss0.015.pt')
@package('/huge/deep/runs/run_Sep16-YB-ft1hr24-4hr48-6hr96-rand_20240916-153935/model_epoch0_iter7495_step1499_loss0.011.pt')
def get_yamahabachelor():
    """We fucked up by overwriting this getter a few times and having it grab different weights.
       Beware which version of the getter you are assuming"""
    extra = ['logtp', '15_msnswrf', '45_tcc']
    input = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=[1,6],
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
        window_size=(3,5,7)))
    return model
    yamaha = get_yamaha()
    bachelor = get_serp3bachelor()
    model.load_state_dict(bachelor.state_dict(),strict=False)
    model.processors['1'].load_state_dict(yamaha.processors['1'].state_dict())
    def assert_eqaul_models(model1,model2):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert p1.data.ne(p2.data).sum() == 0, "models do not match"
    #assert_eqaul_models(yamaha.processors['1'],bachelor.processors['6'])
    return model

@package(load_path='resave')
def get_yamahabachelor0():
    """This loads serp3bachelor + yamaha without any finetuning"""
    extra = ['logtp', '15_msnswrf', '45_tcc']
    input = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=[1,6],
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
        window_size=(3,5,7)))
    yamaha = get_yamaha()
    bachelor = get_serp3bachelor()
    model.load_state_dict(bachelor.state_dict(),strict=False)
    model.processors['1'].load_state_dict(yamaha.processors['1'].state_dict())
    return model

@package('/huge/deep/runs/run_Sep16-YB-ft1hr24-4hr48-6hr96-rand_20240916-153935/model_epoch0_iter7495_step1499_loss0.011.pt')
def get_yamahabachelor3():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    input = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=[1,6],
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
        window_size=(3,5,7)))
    return model

@package('/huge/deep/runs/run_Sep18-YB-ft-rand144hr_20240918-181544/model_epoch0_iter7996_step1999_loss0.018.pt')
def get_yamahabachelor4():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    input = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=[1,6],
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
        window_size=(3,5,7)))
    return model

@package('/huge/deep/runs/run_Sep20-YB-ft-serp3-rand144hr_20240920-113243/model_epoch0_iter13996_step3499_loss0.010.pt')
def get_yamahabachelor5():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    input = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=[1,6],
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
        window_size=(3,5,7)))
    return model

@package('/huge/deep/runs/run_Sep26-YBft-rand144hr-long_20240927-163327/model_epoch0_iter39996_step9999_loss0.008.pt')
def get_yamahabachelor6():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    input = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=[1,6],
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
        window_size=(3,5,7)))
    return model

@package('/huge/deep/runs/run_Oct14-naive-s2s-longer_20241016-140459/model_epoch1_iter184794_step30799_loss0.021.pt')
def get_naives2s():
    extra = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, 
            dims_per_head=32,
            processor_dt=24,
            neorad=True,
            window_size=(5,7,7)))
    return model

@package('/huge/deep/runs/run_Oct10-naive-got-this2s_20241011-022032/model_epoch0_iter111594_step18599_loss0.023.pt')
def get_naives2s_short():
    extra = ['45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, 
            dims_per_head=32,
            processor_dt=24,
            neorad=True,
            window_size=(5,7,7)))
    return model

@package('/huge/deep/runs/run_Oct23-cas2sandra_20241023-185915/model_epoch1_iter209994_step34999_loss0.025.pt')
def get_cas2sandra():
    extra_in_out = ['45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h']
    extra_out_only = ['201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(4,8,8),
            hidden_dim=768,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, 
            dims_per_head=32,
            processor_dt=24,
            neorad=True,
            window_size=(5,7,7)))
    return model

@package('/huge/deep/runs/run_Nov12-2crass2spurious_20241113-012030/model_epoch0_iter122994_step20499_loss0.008.pt')
def get_cras2s():
    extra_in_out = ['45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h'] #msnswrf
    extra_out_only = [] #'142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            checkpointfn=None,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, 
            dims_per_head=32,
            processor_dt=24,
            neorad=True,
            window_size=(5,7,7)))
    return model

@package('/huge/deep/runs/run_Nov12-2crass2spurious_20241113-012030/model_epoch0_iter31194_step5199_loss0.038.pt')
def get_cras2s_step5200():
    extra_in_out = ['45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h'] #msnswrf
    extra_out_only = [] #'142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            checkpointfn=None,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, 
            dims_per_head=32,
            processor_dt=24,
            neorad=True,
            window_size=(5,7,7)))
    return model

@package('/huge/deep/runs/run_Nov12-2crass2spurious_20241113-012030/model_epoch0_iter57594_step9599_loss0.012.pt')
def get_cras2s_step9600():
    extra_in_out = ['45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h'] #msnswrf
    extra_out_only = [] #'142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            checkpointfn=None,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, 
            dims_per_head=32,
            processor_dt=24,
            neorad=True,
            window_size=(5,7,7)))
    return model

@package('/huge/deep/runs/run_Nov12-2crass2spurious_20241113-012030/model_epoch0_iter92394_step15399_loss0.010.pt')
def get_cras2s_step15400():
    extra_in_out = ['45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h'] #msnswrf
    extra_out_only = [] #'142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_in_out, extra_sfc_pad=len(extra_out_only), input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    model = ForecastStep3D(
        ForecastStepConfig(
            [imesh],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            checkpointfn=None,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, 
            dims_per_head=32,
            processor_dt=24,
            neorad=True,
            window_size=(5,7,7)))
    return model

@package('/huge/deep/runs/run_Dec11-thor_20241211-172115/model_epoch7_iter39996_step9999_loss0.003.pt')
def get_thor():
    extra = []#'45_tcc', '034_sstk', '168_2d', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h'] #msnswrf
    mesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_ecm1)
    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvDecoder
    from model_latlon.encoder import SimpleConvEncoder
    conf = ForecastModelConfig(
        [mesh],
        encdec_tr_depth=4,
        oldpr=True,
        latent_size=1024,
        pr_dims = [48, 192, 512],
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
        checkpoint_type="none",
    )
    model = ForecastModel(mesh,conf,
                          encoders=[SimpleConvEncoder(mesh,conf)],
                          decoders=[SimpleConvDecoder(mesh,conf)])
    return model

@package('/huge/deep/runs/run_Dec20-wodan_20241220-020040/model_epoch7_iter39996_step9999_loss0.002.pt')
def get_wodan():
    extra = ['45_tcc', '034_sstk', '168_2d', '179_ttr', '136_tcw', '137_tcwv']
    mesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_ecm1)
    from model_latlon.top import ForecastModel, ForecastModelConfig, StackedConvPlusDecoder
    from model_latlon.encoder import SimpleConvEncoder
    conf = ForecastModelConfig(
        [mesh],
        encdec_tr_depth=4,
        oldpr=True,
        latent_size=512,
        pr_dims = [48, 192, 512],
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
        checkpoint_type="none",
    )
    model = ForecastModel(mesh,conf,
                          encoders=[SimpleConvEncoder(mesh,conf)],
                          decoders=[StackedConvPlusDecoder(mesh,conf)])
    return model

@package('/huge/deep/runs/run_Dec25-freyja_20241225-035807/model_epoch10_iter59994_step9999_loss0.011.pt')
def get_freyja():
    extra = ['45_tcc', '034_sstk', '168_2d', '179_ttr', '136_tcw', '137_tcwv',
             '142_lsp', '143_cp'] 
    imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=[], input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_ecm1)
    from model_latlon.top import ForecastModel, ForecastModelConfig, StackedConvPlusDecoder
    from model_latlon.encoder import SimpleConvEncoder
    conf = ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        encdec_tr_depth=4,
        oldpr=True,
        latent_size=512,
        pr_dims = [48, 192, 512],
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
        checkpoint_type="none",
    )
    model = ForecastModel(imesh,conf,
                          encoders=[SimpleConvEncoder(imesh,conf)],
                          decoders=[StackedConvPlusDecoder(omesh,conf)])
    return model

@package('/huge/deep/runs/run_Dec30-tyr_20241230-102240/model_epoch15_iter85794_step14299_loss0.006.pt')
def get_tyr():
    extra_in = ['45_tcc', '034_sstk', '168_2d', '136_tcw', '137_tcwv']
    extra_out = extra_in + ['179_ttr', '142_lsp', '143_cp', 'logtp', '201_mx2t', '202_mn2t']
    imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_in, input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_out, input_levels=levels_medium, levels=levels_smol)
    from model_latlon.top import ForecastModel, ForecastModelConfig
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.decoder import SimpleConvDecoder
    conf = ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        encdec_tr_depth=4,
        simple_decoder_patch_size=(1,8,8),
        latent_size=768,
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
        checkpoint_type="none",
    )
    model = ForecastModel(imesh,conf,
                          encoders=[SimpleConvEncoder(imesh,conf)],
                          decoders=[SimpleConvDecoder(omesh,conf)])
    return model

@package('/huge/deep/runs/run_Dec30-tyr_20241230-102240/model_epoch15_iter85794_step14299_loss0.006.pt')
def get_fenrir():
    extra_in = ['45_tcc', '034_sstk', '168_2d', '136_tcw', '137_tcwv']
    extra_out = extra_in + ['179_ttr', '142_lsp', '143_cp', 'logtp', '201_mx2t', '202_mn2t']
    #imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=[], input_levels=levels_medium)
    imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_in, input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_out, input_levels=levels_medium, levels=levels_smol)
    from model_latlon.top import ForecastModel, ForecastModelConfig
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.decoder import SimpleConvDecoder
    conf = ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        encdec_tr_depth=4,
        simple_decoder_patch_size=(1,8,8),
        latent_size=768,
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
        checkpoint_type="none",
    )
    model = ForecastModel(imesh,conf,
                          encoders=[SimpleConvEncoder(imesh,conf)],
                          decoders=[SimpleConvDecoder(omesh,conf)])
    return model

@package('/huge/deep/runs/run_Jan6-fenrir_20250106-184214/model_epoch11_iter64995_step12999_loss0.006.pt')
def get_fenrir():
    extra_in = ['45_tcc', '034_sstk', '168_2d', '136_tcw', '137_tcwv']
    extra_out = extra_in + ['179_ttr', '142_lsp', '143_cp', 'logtp', '201_mx2t', '202_mn2t']
    imesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_in, input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5_daily-28',extra_sfc_vars=extra_out, input_levels=levels_medium, levels=levels_smol)
    from model_latlon.top import ForecastModel, ForecastModelConfig
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.decoder import SimpleConvDecoder
    conf = ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        encdec_tr_depth=4,
        simple_decoder_patch_size=(1,8,8),
        latent_size=768,
        pr_depth = [10],
        tr_embedding='rotary',
        processor_dts = [24],
        checkpoint_type="none",
    )
    model = ForecastModel(conf,
                          encoders=[SimpleConvEncoder(imesh,conf)],
                          decoders=[SimpleConvDecoder(omesh,conf)])
    return model

@package('/huge/deep/runs/run_Oct22-stale_20241022-101004/model_epoch3_iter81196_step20299_loss0.120.pt')
def get_stale():
    mesh = meshes.LatLonGrid(source='era5-28', input_levels=levels_medium, levels=levels_medium)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh],
            outputs=[mesh],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(4,8,8),
            hidden_dim=512,
            enc_swin_depth=6, 
            dec_swin_depth=6,
            proc_swin_depth=6, 
            dims_per_head=32,
            processor_dt=6,
            neorad=True,
            window_size=(5,7,7)))
    return model

@package(load_path='resave')
#@package()
def get_rtyamahabachelor():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny,levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    mesh3 = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2], 
            outputs=[omesh], 
            rollout_reencoder=True,
            rollout_inputs=[mesh3],
            sincos=True, 
            padded_lon=True,
            output_deltas=False, 
            Transformer=SlideLayer3D, 
            checkpointfn=None,
            patch_size=(5,8,8), 
            hidden_dim=896, 
            enc_swin_depth=4, 
            dec_swin_depth=4, 
            proc_swin_depth=8,
            dims_per_head=32, 
            processor_dt=[1,6], 
            decoder_reinput_initial=True, 
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    
    #print(model)
    #return model
    rt = get_rtbachelor()
    yamaha = get_yamaha()
    bachelor = get_serp3bachelor()
    model.load_state_dict(bachelor.state_dict(),strict=False)
    model.encoders.load_state_dict(rt.encoders.state_dict())
    model.rollout_reencoder.load_state_dict(bachelor.encoder.state_dict())
    model.processors['1'].load_state_dict(yamaha.processors['1'].state_dict())
    model.processors['6'].load_state_dict(bachelor.processors['6'].state_dict())
    return model

@package(load_path='resave')
#@package()
def get_rtyamahabachelor2():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny,levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    mesh3 = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2],
            outputs=[omesh],
            rollout_reencoder=True,
            rollout_inputs=[mesh3],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=[1,6],
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    sd_path = '/huge/deep/runs/run_Sep16-YB-1hr24-rtft_20240916-120929/model_epoch6_iter5997_step1999_loss0.065.pt'
    model.load_state_dict(torch.load(sd_path)["model_state_dict"], strict=False)
    bachelor = get_serp3bachelor()
    model.rollout_reencoder.load_state_dict(bachelor.encoder.state_dict())
    return model

@package(load_path='resave')
def get_rtyamahabachelor3():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny,levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    mesh3 = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2],
            outputs=[omesh],
            rollout_reencoder=True,
            rollout_inputs=[mesh3],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=[1,6],
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    # sd_path = '/huge/deep/runs/run_Sep17-YB-rand96hr-rtft_20240917-205654/model_epoch13_iter13194_step2199_loss0.070.pt'
    # model.load_state_dict(torch.load(sd_path)["model_state_dict"], strict=False)
    # bachelor = get_serp3bachelor()
    # model.rollout_reencoder.load_state_dict(bachelor.encoder.state_dict())
    return model

@package(load_path='resave')
def get_rtyamahabachelor4():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(source='hres-13',extra_sfc_pad=3, input_levels=levels_tiny,levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    mesh3 = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2],
            outputs=[omesh],
            rollout_reencoder=True,
            rollout_inputs=[mesh3],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=[1,6],
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    # sd_path = '/huge/deep/runs/run_Sep20-YB-rand144hr-rtft_20240920-113320/model_epoch13_iter13194_step2199_loss0.058.pt'
    # model.load_state_dict(torch.load(sd_path)["model_state_dict"], strict=False)
    # bachelor = get_serp3bachelor()
    # model.rollout_reencoder.load_state_dict(bachelor.encoder.state_dict())
    return model

@package(load_path='resave')
def get_rtyamahabachelor5():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny,levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    mesh3 = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2],
            outputs=[omesh],
            rollout_reencoder=True,
            rollout_inputs=[mesh3],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=[1,6],
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    # sd_path = '/huge/deep/runs/run_Sep24-YB-serp3rand144hr-rtft_20240924-181637/model_epoch12_iter11994_step1999_loss0.058.pt'
    # model.load_state_dict(torch.load(sd_path)["model_state_dict"], strict=False)
    # yb = get_yamahabachelor5()
    # model.rollout_reencoder.load_state_dict(yb.encoder.state_dict())
    return model

@package(load_path='resave')
def get_rtyblong():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_pad=3, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(source='hres-13', extra_sfc_pad=3, input_levels=levels_tiny,levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    mesh3 = meshes.LatLonGrid(source='era5-28', extra_sfc_pad=3, input_levels=levels_medium, levels=levels_joank)
    model = ForecastStep3D(
        ForecastStepConfig(
            [mesh1, mesh2],
            outputs=[omesh],
            rollout_reencoder=True,
            rollout_inputs=[mesh3],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=896,
            enc_swin_depth=4,
            dec_swin_depth=4,
            proc_swin_depth=8,
            dims_per_head=32,
            processor_dt=[1,6],
            decoder_reinput_initial=True,
            decoder_reinput_size=96,
            neorad=True,
            parallel_encoders= True, 
        window_size=(3,5,7)))
    # sd_path = '/huge/deep/runs/run_Oct7-rtyb-rand144hr-long_20241007-175713/model_epoch8_iter7996_step1999_loss0.011.pt'
    # model.load_state_dict(torch.load(sd_path)["model_state_dict"], strict=False)
    # yb = get_yamahabachelor6()
    # model.rollout_reencoder.load_state_dict(yb.encoder.state_dict())
    return model


def get_ducati():
    extra_input = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp', '143_cp', '201_mx2t', '202_mn2t', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    input = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_input, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

    model = ForecastStep3D(
        ForecastStepConfig(
            [input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=1536,
            enc_swin_depth=0,
            dec_swin_depth=0,
            proc_swin_depth=2,
            dims_per_head=32,
            processor_dt=1,
            decoder_reinput_initial=False,
            neorad=True,
        window_size=(3,5,7)))
    return model


#@package('/huge/deep/runs/run_Oct29-doctorate-ft_20240907-012345/model_epoch0_iter6_step1_loss0.066.pt')
#@package('/huge/deep/runs/run_Oct29-doctorate-ft_20241030-112206/model_epoch0_iter31188_step2599_loss0.013.pt')
#@package('/huge/deep/runs/run_Nov6-doctorate-perpetual-stew_20241030-112206/model_epoch0_iter55188_step6599_loss0.020.pt')
#@package('/huge/deep/runs/run_Nov6-doctorate-perpetual-stew_20241030-112206/model_epoch0_iter54288_step6449_loss0.006.pt')
#@package('/huge/deep/runs/run_Nov6-doctorate-perpetual-stew_20241030-112206/model_epoch0_iter35088_step3249_loss0.017.pt')
#@package('/huge/deep/runs/run_Nov12-doctorate-perpetual-stew2_20241030-112206/model_epoch0_iter294_step49_loss0.022.pt')
#@package('/huge/deep/runs/run_Nov12-doctorate-perpetual-stew2_20241030-112206/model_epoch0_iter894_step149_loss0.018.pt')
#@package('/huge/deep/runs/run_Nov12-doctorate-perpetual-stew3_20241030-112206/model_epoch0_iter894_step149_loss0.014.pt')
@package('/huge/deep/runs/run_Nov12-doctorate-perpetual-stew3_20241030-112206/model_epoch0_iter4794_step799_loss0.019.pt')
def get_latentdoctorate():
    #extra = ['logtp', '15_msnswrf', '45_tcc']
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    #extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    #extra_all = extra_in_out + extra_out_only

    extra_input = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    input = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_input, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

    model = ForecastStep3D(
        ForecastStepConfig(
            inputs=[input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=1536, #1408,
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            dims_per_head=32,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
            name='doctorate',
        window_size=(3,5,7)))


    #model = ForecastStepSwin3D(ForecastStepConfig(inputs=[input], outputs=[output], sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpoint_every=np.nan, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8),
    #                                          hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32,
    #                                          processor_dt=6, use_matepoint=False, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,7)))
    return model


#@package('/huge/deep/runs/run_Oct29-doctorate-ft_20241030-112206/model_epoch0_iter31188_step2599_loss0.013.pt')
#@package('/huge/deep/runs/run_Nov1-ducati_20241101-185531/model_epoch1_iter34794_step5799_loss0.013.pt')
#@package('/huge/deep/runs/run_Nov1-ducati_20241103-120402/model_epoch0_iter8994_step1499_loss0.015.pt')
#@package('/huge/deep/runs/run_Nov1-ducati_20241103-120402/model_epoch0_iter24594_step4099_loss0.013.pt')
@package('/huge/deep/runs/run_Nov1-ducati_20241103-120402/model_epoch0_iter77394_step12899_loss0.016.pt')
def get_ducatidoctorate():
    extra_input = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp', '143_cp', '201_mx2t', '202_mn2t', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    input = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_input, input_levels=levels_medium, levels=levels_joank)
    output = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

    model = ForecastStep3D(
        ForecastStepConfig(
            [input],
            outputs=[output],
            sincos=True,
            padded_lon=True,
            output_deltas=False,
            Transformer=SlideLayer3D,
            patch_size=(5,8,8),
            hidden_dim=1536,
            checkpointfn=None,
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            proc_depths=[2,6],#[2,6],
            dims_per_head=32,
            processor_dt=[1,3],#[1,3],
            decoder_reinput_initial=False,
            neorad=True,
            name='doctorate',
        window_size=(3,5,7)))

    """
    ducati = get_ducati()
    doctorate = get_latentdoctorate()
    model.load_state_dict(doctorate.state_dict(),strict=False)
    #print("aa", list(model.processors['1'].state_dict().keys()))
    #print("bb", list(ducati.processors['1'].state_dict().keys()))
    model.processors['1'].load_state_dict(ducati.processors['1'].state_dict())
    """
    return model

#@package('/huge/deep/runs/run_Nov2-doctor-neohresgfs-blend_20241103-011101/model_epoch62_iter29994_step4999_loss0.059.pt')
#@package('/huge/deep/runs/run_Nov2-doctor-neohresgfs-blend_20241103-011101/model_epoch8_iter4194_step699_loss0.069.pt')
@package('/huge/deep/runs/run_Nov2-doctor-neohresgfs-blend_20241104-182658/model_epoch12_iter11394_step1899_loss0.060.pt')
def get_gfsblenddoctor():
    #extra = ['logtp', '15_msnswrf', '45_tcc']
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    #extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    #extra_all = extra_in_out + extra_out_only

    extra_input = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    #input = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_input, input_levels=levels_medium, levels=levels_joank)
    #output = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

    mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_vars=extra_input, input_levels=levels_gfs, levels=levels_joank)
    mesh2 = meshes.LatLonGrid(source='neohres-20', extra_sfc_vars=extra_input, input_levels=levels_hres,levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

    model = ForecastStep3D(
        ForecastStepConfig(
            inputs=[mesh1, mesh2],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=1536, #1408,
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            dims_per_head=32,
            parallel_encoders=True,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
            name='doctorate',
        window_size=(3,5,7)))


    #model = ForecastStepSwin3D(ForecastStepConfig(inputs=[input], outputs=[output], sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpoint_every=np.nan, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8),
    #                                          hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32,
    #                                          processor_dt=6, use_matepoint=False, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,7)))
    return model

#@package('/huge/deep/runs/run_Nov2-doctor-neohresonly_20241104-134946/model_epoch2_iter2394_step399_loss0.546.pt')
@package('/huge/deep/runs/run_Nov2-doctor-neohresonly-neo_20241104-231747/model_epoch7_iter8394_step1399_loss0.065.pt')
def get_neohresdoctor():
    #extra = ['logtp', '15_msnswrf', '45_tcc']
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_pad=3)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #input = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, is_output_only=True)
    #output = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra,is_output_only=True)
    #extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    #extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    #extra_all = extra_in_out + extra_out_only

    extra_input = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             'zeropad', 'zeropad', 'zeropad', 'zeropad', '246_100u', '247_100v'] # last 4 are output only

    extra_output = ['15_msnswrf', '45_tcc', '034_sstk', '168_2d', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h', '246_100u', '247_100v']#, '246_100u', '247_100v'] # last 4 are output only

    #input = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_input, input_levels=levels_medium, levels=levels_joank)
    #output = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

    #mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_vars=extra_input, input_levels=levels_gfs, levels=levels_joank)
    #mesh2 = meshes.LatLonGrid(source='neohres-20', extra_sfc_vars=extra_input, input_levels=levels_hres,levels=levels_joank)
    mesh2 = meshes.LatLonGrid(source='neohres-20', extra_sfc_vars=extra_input, input_levels=levels_hres, intermediate_levels=[levels_ecm1], levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

    model = ForecastStep3D(
        ForecastStepConfig(
            inputs=[mesh2],
            outputs=[omesh],
            sincos=True,
            padded_lon=True,
            Transformer=SlideLayer3D,
            checkpointfn=None,
            patch_size=(5,8,8),
            hidden_dim=1536, #1408,
            enc_swin_depth=6,
            dec_swin_depth=6,
            proc_swin_depth=6,
            dims_per_head=32,
            parallel_encoders=False,
            processor_dt=3,
            output_deltas=False,
            decoder_reinput_initial=False,
            neorad=True,
            name='doctorate',
        window_size=(3,5,7)))


    #model = ForecastStepSwin3D(ForecastStepConfig(inputs=[input], outputs=[output], sincos=True, padded_lon=True, Transformer=SlideLayer3D, checkpoint_every=np.nan, checkpointfn=matepoint.checkpoint,patch_size=(5,8,8),
    #                                          hidden_dim=896, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, timesteps=timesteps, dims_per_head=32,
    #                                          processor_dt=6, use_matepoint=False, output_deltas=False, decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True, window_size=(3,5,7)))
    return model



@package(load_path='/huge/deep/runs_diffusion3/run_Nov23-96-24hr-linear-200k_20241123-123516/model_epoch1_iter202999_step202999_loss0.108.pt',strict=False)
def get_brownian():
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=['logtp', '15_msnswrf', '45_tcc'], input_levels=levels_medium, levels=levels_joank)
    bachelor = get_serp3bachelor(no_load=True)
    from diffusion.model import UNet
    diffuser = UNet(
        in_channels=mesh.n_sfc_vars,
        out_channels=mesh.n_sfc_vars,
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256
    )
    model = ForecastCombinedDiffusion(forecaster=bachelor,diffuser=diffuser,deltas=False)
    return model

@package(load_path='/huge/deep/runs_diffusion3/run_Nov23-96-24hr-cosine-200k_20241123-120150/model_epoch1_iter202999_step202999_loss0.001.pt',strict=False)
def get_cosbrownian():
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=['logtp', '15_msnswrf', '45_tcc'], input_levels=levels_medium, levels=levels_joank)
    bachelor = get_serp3bachelor(no_load=True)
    from diffusion.model import UNet
    diffuser = UNet(
        in_channels=mesh.n_sfc_vars,
        out_channels=mesh.n_sfc_vars,
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256
    )
    model = ForecastCombinedDiffusion(forecaster=bachelor,diffuser=diffuser,schedule='cosine',deltas=False)
    return model


@package(load_path='/huge/deep/runs_diffusion3/run_Nov30-deltas-100k-2g-normed_20241203-012644/model_epoch0_iter109998_step54999_loss0.044.pt',strict=False)
def get_deltasbrownian():
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=['logtp', '15_msnswrf', '45_tcc'], input_levels=levels_medium, levels=levels_joank)
    bachelor = get_serp3bachelor(no_load=True)
    from diffusion.model import UNet
    diffuser = UNet(
        in_channels=mesh.n_sfc_vars,
        out_channels=mesh.n_sfc_vars,
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        conditioning_channels=512
    )
    model = ForecastCombinedDiffusion(forecaster=bachelor,diffuser=diffuser,schedule='linear')
    return model

@package("/huge/deep/runs_diffusion3/run_Dec5-gibbs-multitimestep_20241205-145424/model_epoch4_iter587997_step195999_loss0.066.pt")
def get_multigibbs():
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=['logtp', '15_msnswrf', '45_tcc'], input_levels=levels_medium, levels=levels_joank)
    bachelor = get_serp3bachelor(no_load=True)
    from diffusion.model import UNet
    diffuser = UNet(
        in_channels=mesh.n_sfc_vars*2,
        out_channels=mesh.n_sfc_vars,
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        conditioning_channels=512
    )
    model = ForecastCombinedDiffusion(forecaster=bachelor,diffuser=diffuser,schedule='linear',append_input=True)
    return model

@package("/huge/deep/runs/run_Nov22-joansucks-oldpr_20241122-195813/model_epoch1_iter250794_step41799_loss0.020.pt",strict=False)
def get_joansucks():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk',
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    import model_latlon.top as top# import ForecastModel, ForecastModelConfig
    conf = top.ForecastModelConfig([imesh],
        outputs=[omesh],
        latent_size=1024,
        pr_dims=[48, 192, 512],
        affine=True,
        encdec_tr_depth=2,
        oldpr=True,
        checkpoint_type="none"
    )
    conf.use_pole_convs = False
    from model_latlon.encoder import ResConvEncoder
    encoder = ResConvEncoder(imesh,conf)
    decoder = top.ResConvDecoder(omesh,conf,fuck_the_poles=True)
    model = top.ForecastModel(conf,encoders=[encoder],decoders=[decoder])
    return model


@package("/huge/deep/runs/run_Nov29-rotary-resume_20241129-232324/model_epoch1_iter212297_step29699_loss0.035.pt")#,strict=False)
def get_rotary():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)

    from model_latlon.top import ForecastModel, ForecastModelConfig, SimpleConvEncoder, SimpleConvDecoder, ResConvEncoder, ResConvDecoder
    conf = ForecastModelConfig(
        [imesh],
        outputs=[omesh],
        latent_size = 1280,
        pr_dims = [48, 192, 512],
        affine = True,
        pr_depth = 10,
        encdec_tr_depth = 4,
        oldpr = True,
        tr_embedding = 'rotary',
    )
    encoder = ResConvEncoder(imesh,conf)
    decoder = ResConvDecoder(omesh,conf)
    model = ForecastModel(imesh,conf,encoders=[encoder],decoders=[decoder])
    return model

@package('/huge/deep/runs/run_Dec3-regionaldecoder-test_20241203-191255/model_epoch0_iter11198_step5599_loss0.188.pt')
def get_regionaldecodertest():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
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
    return model

@package("/huge/deep/runs/run_Dec6-joanlatentsucks_20241122-195813/model_epoch1_iter98394_step16399_loss0.010.pt")
def get_joanlatentsucks():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    import model_latlon.top as top# import ForecastModel, ForecastModelConfig
    conf = top.ForecastModelConfig(inputs=[imesh], outputs=[omesh])
    conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    conf.affine = True
    conf.encdec_tr_depth = 2
    conf.oldpr = True
    conf.update()
    encoder = top.ResConvEncoder(imesh,conf)
    decoder = top.ResConvDecoder(omesh,conf)

    model = top.ForecastModel(imesh,conf, encoders=[encoder], decoders=[decoder])
    return model

@package("/huge/deep/runs/run_Dec6-joanlatentsucks_20241122-195813/model_epoch1_iter98394_step16399_loss0.010.pt")
def get_joanbikessucks():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    import model_latlon.top as top# import ForecastModel, ForecastModelConfig
    conf = top.ForecastModelConfig(inputs=[imesh], outputs=[omesh])
    conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    conf.pr_depth = [2, 8]
    conf.processor_dts = [1, 6]
    conf.affine = True
    conf.encdec_tr_depth = 2
    #conf.checkpoint_type = "torch"
    conf.oldpr = True
    conf.update()
    encoder = top.ResConvEncoder(imesh,conf)
    decoder = top.ResConvDecoder(omesh,conf)

    model = top.ForecastModel(imesh,conf, encoders=[encoder], decoders=[decoder])
    return model

#@package("/huge/deep/runs/run_Dec6-joanlatentsucks_20241122-195813/model_epoch1_iter98394_step16399_loss0.010.pt")
@package("/huge/deep/runs/run_Dec16-realtimesucks72_20241221-085921/model_epoch34_iter63594_step10599_loss0.079.pt")
def get_joanrealtimesucks():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    #imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    imesh1 = meshes.LatLonGrid(source='neogfs-25',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_gfs, levels=levels_medium)
    imesh2 = meshes.LatLonGrid(source='neohres-20',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_hres, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    import model_latlon.top as top# import ForecastModel, ForecastModelConfig
    import model_latlon.encoder as encoder# import ForecastModel, ForecastModelConfig
    conf = top.ForecastModelConfig(inputs=[imesh1, imesh2], outputs=[omesh])
    conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    #conf.pr_depth = [2, 8]
    #conf.processor_dts = [1, 6]
    conf.parallel_encoders=True
    conf.encoder_weights=[0.1,0.9]
    conf.affine = True
    conf.encdec_tr_depth = 2
    #conf.checkpoint_type = "torch"
    conf.oldpr = True
    conf.update()
    encoder1 = encoder.ResConvEncoder(imesh1,conf)
    encoder2 = encoder.ResConvEncoder(imesh2,conf)
    decoder = top.ResConvDecoder(omesh,conf)

    model = top.ForecastModel(conf, encoders=[encoder1, encoder2], decoders=[decoder])
    return model

@package("/huge/deep/runs/run_Dec31-pointdecodertest_20241231-141734/model_epoch0_iter91798_step21299_loss0.167.pt")
def get_pointdecodertest():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    from model_latlon.top import ForecastModel, ForecastModelConfig
    from model_latlon.encoder import ResConvEncoder
    from model_latlon.decoder import ResConvDecoder, PointDecoder

    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size=1024,
        pr_dims=[48, 192, 512],
        affine=True,
        encdec_tr_depth=2,
        oldpr=True,
        checkpoint_type="none"
    )
    encoder = ResConvEncoder(imesh,conf)
    era5_dercoder = ResConvDecoder(omesh, conf, fuck_the_poles=True)
    point_decoder = PointDecoder(omesh, conf, hidden_dim=1024)
    model = ForecastModel(imesh, conf, encoders=[encoder], decoders=[era5_dercoder, point_decoder])
    return model

@package("/huge/deep/runs/run_Dec16-realtimesucks-hresonly_20250101-145023/model_epoch44_iter93995_step18799_loss0.084.pt")
def get_joanrealtimesucks_hresonly():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh2 = meshes.LatLonGrid(source='neohres-20',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_hres, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    import model_latlon.top as top# import ForecastModel, ForecastModelConfig
    import model_latlon.encoder as encoder# import ForecastModel, ForecastModelConfig
    conf = top.ForecastModelConfig(inputs=[imesh2], outputs=[omesh])
    conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    #conf.pr_depth = [2, 8]
    #conf.processor_dts = [1, 6]
    conf.parallel_encoders=False
    conf.encoder_weights=[0.1,0.9]
    conf.affine = True
    conf.encdec_tr_depth = 2
    #conf.checkpoint_type = "torch"
    conf.oldpr = True
    conf.use_pole_convs = False
    conf.update()
    encoder2 = encoder.ResConvEncoder(imesh2,conf)
    decoder = top.ResConvDecoder(omesh,conf)

    model = top.ForecastModel(conf, encoders=[encoder2], decoders=[decoder])
    return model


@package("/huge/deep/runs/run_Dec16-realtimesucks72_20241221-085921/model_epoch27_iter50394_step8399_loss0.079.pt")
def get_joanrealtimesucks72():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh1 = meshes.LatLonGrid(source='neogfs-25',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_gfs, levels=levels_medium)
    imesh2 = meshes.LatLonGrid(source='neohres-20',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_hres, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    #mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_vars=extra_input, input_levels=levels_gfs, levels=levels_joank)
    #mesh2 = meshes.LatLonGrid(source='neohres-20', extra_sfc_vars=extra_input, input_levels=levels_hres,levels=levels_joank)
    #omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)


    import model_latlon.top as top# import ForecastModel, ForecastModelConfig
    conf = top.ForecastModelConfig(inputs=[imesh1, imesh2], outputs=[omesh])
    conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    #conf.pr_depth = [2, 8]
    #conf.processor_dts = [1, 6]
    conf.parallel_encoders=True
    conf.encoder_weights=[0.1,0.9]
    conf.affine = True
    conf.encdec_tr_depth = 2
    conf.checkpoint_type = "none"
    conf.oldpr = True
    conf.update()
    encoder1 = top.ResConvEncoder(imesh1,conf)
    encoder2 = top.ResConvEncoder(imesh2,conf)
    decoder = top.ResConvDecoder(omesh,conf)

    model = top.ForecastModel(imesh1,conf, encoders=[encoder1, encoder2], decoders=[decoder])
    model.name = "joanrealtimesucks"
    return model

#@package("/huge/deep/runs/run_Dec6-joanlatentsucks_20241122-195813/model_epoch1_iter98394_step16399_loss0.010.pt")
@package("/huge/deep/runs/run_Dec16-joanbikes_20241216-083341/model_epoch1_iter182394_step30399_loss0.088.pt")
def get_joanbikessucks():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    import model_latlon.top as top# import ForecastModel, ForecastModelConfig
    conf = top.ForecastModelConfig(inputs=[imesh], outputs=[omesh])
    conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    conf.pr_depth = [2, 8]
    conf.processor_dts = [1, 6]
    conf.affine = True
    conf.encdec_tr_depth = 2
    #conf.checkpoint_type = "torch"
    conf.oldpr = True
    conf.update()
    encoder = top.ResConvEncoder(imesh,conf)
    decoder = top.ResConvDecoder(omesh,conf)

    model = top.ForecastModel(imesh,conf, encoders=[encoder], decoders=[decoder])
    model.name = "joanbikessucks"
    return model


#@package("/huge/deep/runs/run_Dec6-joanlatentsucks_20241122-195813/model_epoch1_iter98394_step16399_loss0.010.pt")
#@package("/huge/deep/runs/run_Dec16-joanbikes_20241216-083341/model_epoch1_iter182394_step30399_loss0.088.pt")
@package('resave')
def get_joancompletelysucks():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh1 = meshes.LatLonGrid(source='neogfs-25',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_gfs, levels=levels_medium)
    imesh2 = meshes.LatLonGrid(source='neohres-20',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_hres, levels=levels_medium)

    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    from model_latlon.encoder import ResConvEncoder
    from model_latlon.decoder import ResConvDecoder
    from model_latlon.top import ForecastModelConfig, ForecastModel
    conf = ForecastModelConfig(inputs=[imesh1, imesh2], outputs=[omesh])
    conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    conf.pr_depth = [2, 8]
    conf.processor_dts = [1, 6]
    conf.affine = True
    conf.encdec_tr_depth = 2
    conf.parallel_encoders=True
    conf.encoder_weights=[0.1,0.9]
    conf.checkpoint_type = "none"
    conf.oldpr = True
    conf.use_pole_convs = False
    conf.update()
    encoder1 = ResConvEncoder(imesh1,conf)
    encoder2 = ResConvEncoder(imesh2,conf)
    decoder = ResConvDecoder(omesh,conf)

    #realtime = get_joanrealtimesucks72()
    #bikes = get_joanbikessucks()

    model = ForecastModel(conf, encoders=[encoder1, encoder2], decoders=[decoder])
    model.name = "joancompletelysucks"
    #model.load_state_dict(bikes.state_dict(),strict=False) # doesn't have encoders[1]
    

    #model.encoders[0].load_state_dict(realtime.encoders[0].state_dict())
    #model.encoders[1].load_state_dict(realtime.encoders[1].state_dict())

    return model

@package("/huge/deep/runs/run_Dec16-realtimesucks72_20241221-085921/model_epoch27_iter50394_step8399_loss0.079.pt")
def get_joanrealtimesucks3():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh1 = meshes.LatLonGrid(source='neogfs-25',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_gfs, levels=levels_medium)
    imesh2 = meshes.LatLonGrid(source='neohres-20',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_hres, levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)

    #mesh1 = meshes.LatLonGrid(source='neogfs-25', extra_sfc_vars=extra_input, input_levels=levels_gfs, levels=levels_joank)
    #mesh2 = meshes.LatLonGrid(source='neohres-20', extra_sfc_vars=extra_input, input_levels=levels_hres,levels=levels_joank)
    #omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)


    import model_latlon.top as top# import ForecastModel, ForecastModelConfig
    conf = top.ForecastModelConfig(inputs=[imesh1, imesh2], outputs=[omesh])
    conf.latent_size = 1024
    conf.pr_dims = [48, 192, 512]
    #conf.pr_depth = [2, 8]
    #conf.processor_dts = [1, 6]
    conf.parallel_encoders=True
    conf.encoder_weights=[0.1,0.9]
    conf.affine = True
    conf.encdec_tr_depth = 2
    conf.checkpoint_type = "none"
    conf.oldpr = True
    conf.update()
    encoder1 = top.ResConvEncoder(imesh1,conf)
    encoder2 = top.ResConvEncoder(imesh2,conf)
    decoder = top.ResConvDecoder(omesh,conf)

    model = top.ForecastModel(imesh1,conf, encoders=[encoder1, encoder2], decoders=[decoder])
    return model

#@package("/huge/deep/runs/run_Jan18-graphbs-tinyyolo_20250119-000210/model_epoch2_iter163495_step32699_loss0.332.pt")
#@package("/huge/deep/runs/run_Jan18-graphbs-tinyyolo_20250119-000210/model_epoch2_iter183495_step36699_loss0.398.pt")
#@package("/huge/deep/runs/run_Jan18-graphbs-tinyyolo_20250119-000210/model_epoch3_iter195495_step39099_loss0.278.pt")
#@package("/huge/deep/runs/run_Jan18-graphbs-tinyyolo_20250119-000210/model_epoch3_iter203995_step40799_loss0.270.pt")
#@package("/huge/deep/runs/run_Jan18-graphbs-tinyyolo_20250119-000210/model_epoch3_iter206995_step41399_loss0.341.pt")
@package("/huge/deep/runs/run_Jan18-graphbs-tinyyolo_20250119-000210/model_epoch3_iter224995_step44999_loss0.201.pt")
def get_graphbs_tinyyolo():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

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
    conf.checkpoint_type = "none"
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = SimpleHealDecoder(omesh, conf, compile=False)
    model = ForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])

    return model

#@package("/huge/deep/runs/run_Jan18-graphbs-yolo_20250118-234336/model_epoch1_iter240794_step40199_loss0.221.pt")
#@package("/huge/deep/runs/run_Jan18-graphbs-yolo_20250118-234336/model_epoch1_iter253394_step42299_loss0.305.pt")
@package("/huge/deep/runs/run_Jan18-graphbs-yolo_20250118-234336/model_epoch2_iter267194_step44599_loss0.212.pt")
def get_graphbs_yolo():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

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
    conf.checkpoint_type = "none"
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = SimpleHealDecoder(omesh, conf, compile=False)
    model = ForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])

    return model

@package("/huge/deep/runs/run_Jan16-operational_stackedconvplus_finetune_20250116-103413/model_epoch1_iter191988_step15999_loss0.025.pt")
def get_operational_stackedconvplus_finetime():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h',
             'tc-maxws', 'tc-minp'] # New TC outputs!

    hres_mesh = meshes.LatLonGrid(source='neohres-20',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_hres, levels=levels_ecm1)
    gfs_mesh = meshes.LatLonGrid(source='neogfs-25',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_gfs, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    
    import model_latlon.top as top
    conf = top.ForecastModelConfig(
        [hres_mesh, gfs_mesh],
        outputs=[omesh],
        latent_size = 1440, # Highly composite number, larger latent space (was 1280)
        affine = True,
        pr_depth = [8],
        encdec_tr_depth = 3, # More Transformers! (was 2)
        tr_embedding = 'rotary',
        patch_size = (4,8,8),
        use_pole_convs = False,
        parallel_encoders = True,
        encoder_weights = [0.9, 0.1],
    )
    
    import model_latlon.encoder as encoders
    import model_latlon.decoder as decoders
    encoder_hres = encoders.SimpleConvEncoder(hres_mesh, conf)
    encoder_gfs = encoders.SimpleConvEncoder(gfs_mesh, conf)
    decoder = decoders.StackedConvPlusDecoder(omesh, conf)
    model = top.ForecastModel(
        conf,
        encoders=[encoder_hres, encoder_gfs],
        decoders=[decoder]
    )
    
    return model

@package("/huge/deep/runs/run_Jan28-graphbs-deepmlp_20250128-210311/model_epoch1_iter205495_step41099_loss0.358.pt")
def get_graphbs_deepmlp():
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_joank)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)

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
    hmesh = HealMesh(8)
    hlmesh = HealLatentMesh(depth=5, D=6, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.checkpoint_type = "none"
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    dec = GNNyHealDecoder(omesh, conf, compile=False)
    model = ForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])

    return model


if __name__ == "__main__":
    #get_neoquadripede()
    #get_tiny()
    #get_rtyamahabachelor3()#resave=True)
    #get_yamahabachelor()#resave=True)
    #get_rtyamahabachelor4()#resave=True)
    #get_rtyamahabachelor5()#resave=True)
    #get_rtyblong()#resave=True)
    #get_brownian()
    #get_joancompletelysucks()#resave=True)
    get_graphbs_deepmlp()
    print('done')
