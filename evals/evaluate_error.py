import torch

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    #torch.multiprocessing.freeze_support()
    torch.set_grad_enabled(False)

import pickle
from utils import *
from eval import unnorm_output, unnorm, unnorm_output_partial, compute_errors
from data import *
from model_latlon_3d import *
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader    
import json
import hashlib
import base64
import boto3


def save_metadata(js, hash, path, upload_metadata=False):
    metapath = os.path.join(os.path.dirname(path), f'meta.{hash}.json')
    if '/ens' in path or '/det/' in path:
        rootpath = '/'.join(path.split('/')[:-3])
        metapath = os.path.join(rootpath, f'meta.{hash}.json')
    else:
        metapath = os.path.join(os.path.dirname(path), f'meta.{hash}.json')
    if os.path.exists(metapath):
        with open(metapath,'r') as f:
            js2 = f.read()
        assert js == js2, "metadata mismatch"
    else:    
        with open(metapath,'w') as f:
            f.write(js)

    if upload_metadata:
        S3_CLIENT = boto3.client('s3')
        BUCKET = 'wb-dlnwp'
        s3_path = '/'.join(metapath.split('/')[-2:])
        try:
            S3_CLIENT.head_object(Bucket=BUCKET, Key=s3_path)
        except S3_CLIENT.exceptions.ClientError:
            print(f'File {metapath} does not exist on S3')
            S3_CLIENT.upload_file(metapath, BUCKET, s3_path)
            print(f'File {metapath} uploaded to {s3_path}')

def save_instance(x,path,mesh, model_name,downsample_levels=False, upload_metadata=False):
    if isinstance(x,torch.Tensor):
        x = x.detach().cpu().numpy()
    if downsample_levels:
        newconf = NeoDatasetConfig(conf_to_copy=mesh.config,levels=levels_ecm2)
        newmesh = type(mesh)(newconf)
        wh_levnew = [mesh.config.levels.index(x) for x in levels_ecm2]
        xshape_new = list(x.shape[:-1]) + [newmesh.n_vars]
        xnew = np.zeros(xshape_new,dtype=x.dtype)
        for i,j in enumerate(wh_levnew):
            xnew[...,i*mesh.n_pr_vars:(i+1)*mesh.n_pr_vars] = x[...,j*mesh.n_pr_vars:(j+1)*mesh.n_pr_vars]
        xnew[...,-mesh.n_sfc:] = x[...,-mesh.n_sfc:]
        x = xnew
        mesh = newmesh
    js,hash = mesh.to_json(model_name)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    save_metadata(js, hash, path, upload_metadata)
    if isinstance(x,torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.shape[0] == 1:
        x = x[0]
    else:
        assert len(x.shape) == 3, "Can not be multi batch"
    filepath= path+f".{hash}.npy"
    print("Saving to", filepath)
    np.save(filepath,x)
    return filepath
    

def to_filename(nix,dt,tags=[], always_plus=False):
    dts = ''
    if dt != 0 or always_plus:
        dts = f'+{dt}'
    tagstr = ''
    if len(tags) > 0:
        tagstr = '.'+'.'.join(tags)
    return f'{get_date_str(nix)}{dts}{tagstr}'

def get_output_filepath(nix, dt, model_type='det'):
    return f'{get_date_str(nix)}/{model_type}/{dt}'

def get_compare_tag(data_config):
    if data_config.output_mesh is None:
        return data_config.mesh.config.source
    else:
        return data_config.mesh.config.source +'->'+data_config.output_mesh.config.source 

def Eval2022adapter(eval_path, dates=None):

    os.makedirs(f'{eval_path}/outputs/',exist_ok=True)
    os.makedirs(f'{eval_path}/errors/',exist_ok=True)

    device = torch.device('cuda:0')
    #print("uh mesh is", mesh, mesh.__dict__)
    if dates is None:
        dates = get_dates((D(2020, 1, 1),D(2020,12,31)))
    print("date range", dates[0], dates[-1])
    #dataset_config = mesh.config
    #mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='gfs-28')
    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='neogfs-25', input_levels=levels_gfs, levels=levels_medium)
    mesh2 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,input_levels=levels_tiny,levels=levels_medium,source='hres-13')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1, mesh2],
                                           outputs = [mesh3],
                                           timesteps=[24],
                                           only_at_z=[0,12],
                                           clamp_output=np.inf,
                                           use_mmap = True,
                                           requested_dates = dates,
                                           ))
    import model_unet

    model_conf = ForecastStepConfig(data.config.inputs, 
                outputs = data.config.outputs,
                patch_size=(4,8,8), 
                hidden_dim=768, 
                enc_swin_depth=0,
                dec_swin_depth=0,
                proc_swin_depth=0,
                adapter_swin_depth=8,
                timesteps=[0], 
                adapter_use_input_bias=True,
                output_deltas = True,
                use_matepoint = False,
                activation = nn.LeakyReLU(negative_slope=0.02, inplace=True)
                )


    adapter_model = model_unet.ForecastStepAdapterConv(model_conf)
    #pp = "/fast/ignored/runs_adapt/run_Dec26-Luigi-l2-to-l1_20231226-151560/model_epoch99_iter95996_loss0.142.pt"
    #pp = "/fast/ignored/runs_adapt/run_Dec26-Luigi-l2-to-l1_20231226-151560/model_epoch136_iter131996_loss0.148.pt"
    pp = "/fast/ignored/runs_adapt/run_Jan22-neohegel_20240123-000636/model_epoch22_iter21596_loss0.080.pt"
    checkpoint = torch.load(pp,map_location='cpu')
    adapter_model.load_state_dict({k.replace("adapter.", ""): v for k, v in checkpoint['model_state_dict'].items()},strict=False)

    with open(f'{eval_path}/model.pickle','rb') as f:
        model = pickle.load(f)

    model.config.mesh.extra_sfc_vars = []
    model.config.neorad = False
    model.do_sub = False
    model.output_deltas = True
    #model.config.decoder_reinput_initial = True
    #for dec in model.decoders:
    #    model.decoders[dec].H = model.decoders[dec].config.hidden_dim

    #print("aaa", model.config.decoder_reinput_size, model.config.decoder_reinput_initial)

    #print("og output", model.config.output_mesh.config.source)
    #print("hey dataset", "mesh", dataset.config.mesh.config.source, "output_mesh", dataset.config.output_mesh.config.source)
    data.check_for_dates()
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=2, collate_fn=default_collate)

    _,stds,means = load_state_norm(mesh3.wh_lev, mesh3,with_means=True)
    adapter_model.output_deltas = True

    adapter_model = adapter_model.to(device)

    model = model.to(device)
    sofar = []
    for i,sample in enumerate(dataloader):
    #for i,sample in enumerate([default_collate([dataset[0]])]):
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            x = sample[0]
            x = [xx.to(device) for xx in x]
            dt = sample_dts(sample)[0]
            print("timestamps", [a[-1] for a in sample])
            #print("hey sample", len(sample), [len(a) for a in sample])
            print("dt is", dt)
            #print("sample[0] is", sample[0], len(sample[0]))
            nix = int(sample[0][-1])
            #print("uhh", [(a.dtype, a.device) for a in x])
            #x = [a.float() for a in x]
            #print("uhh", [(a.dtype, a.device) for a in x])
            y = adapter_model(x,dt=0)
            y.meta = SimpleNamespace(delta_info=0,valid_at=nix)
            #print("hiya y is", y, len(y), len(y[0]))
            _,y = unnorm_output_partial(x[0],y,adapter_model,0)
            #aa = y.float()
            #print(torch.mean(aa, axis=(0,1,2)), "std", torch.std(aa, axis=(0,1,2)))
            #print("uh y", y.shape, y.dtype)
            og = y.clone()
            #print("og", og.shape, "x[-1][0]", x[-1][0].shape)
            y = model([og]+[x[-1]], dt=dt)
            print("Hey uh", y.meta)
            x,y = unnorm_output(og,y,model,dt)
            yt = unnorm(sample[1][0].to(device),mesh3)
            if i % 50 == 0:
                print(f'Saving, x.shape {x.shape}, y.shape {y.shape}')
                save_instance(x,f'{eval_path}/outputs/{to_filename(nix,0)}',mesh3)
                save_instance(y,f'{eval_path}/outputs/{to_filename(nix,dt)}',mesh3)

        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float32):
            rms = compute_errors(y,yt,mesh3)
        sofar.append(rms["129_z_500"])
        print(f'{get_date_str(nix)}, {i}/{len(dataloader)}, z500 {rms["129_z_500"]}', "ohp %.2f pm %.2f"%(np.sqrt(np.mean(np.array(sofar)**2)), np.std(sofar)/(len(sofar)**0.5)))
        with open(f'{eval_path}/errors/{get_date_str(nix)}+{dt}.pickle','wb') as f:
            pickle.dump(rms,f)


def Eval2020(eval_path, dates=None, dt=24):
    #eval_path = '/fast/evaluation/Neoreal_335M/'
    with open(f'{eval_path}/model.pickle','rb') as f:
        model = pickle.load(f)

    os.makedirs(f'{eval_path}/outputs/',exist_ok=True)
    os.makedirs(f'{eval_path}/errors/',exist_ok=True)

    device = torch.device('cuda:0')
    mesh = model.config.mesh
    #print("uh mesh is", mesh, mesh.__dict__)
    if dates is None:
        dates = get_dates((D(2020, 1, 1),D(2020,12,31)))
    print("date range", dates[0], dates[-1])
    #dataset_config = mesh.config
    print("mesh", model.config.mesh.source)
    print("output mesh?", model.config.output_mesh.source)
    output_mesh = model.config.output_mesh
    dataset = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[output_mesh],
                        timesteps=[dt],#model.config.timesteps,
                        requested_dates = dates,
                        use_mmap = True,
                        only_at_z = [0,12],
                        clamp_output = np.inf,
                        ))
    #print("og output", model.config.output_mesh.config.source)
    #print("hey dataset", "mesh", dataset.config.mesh.config.source, "output_mesh", dataset.config.output_mesh.config.source)
    dataset.check_for_dates()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=default_collate)

    _,stds,means = load_state_norm(mesh.wh_lev, mesh,with_means=True)


    model = model.to(device)
    sofar = []
    for i,sample in enumerate(dataloader): # sample = [[input instance, ..., time], [output instance, ..., time] ... ]
    #for i,sample in enumerate([default_collate([dataset[0]])]):
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            x = sample[0]
            x = [xx.to(device) for xx in x]
            dt = sample_dts(sample)[0]
            #print("dt is", dt)
            nix = int(sample[0][1])
            y = model(x,dt=dt) # y is usually a delta
            x,y = unnorm_output(x[0],y,model,dt)
            yt = unnorm(sample[1][0].to(device),mesh) #yt stands y_targets
            if i % 50 == 0:
                print(f'Saving, x.shape {x.shape}, y.shape {y.shape}')
                save_instance(x,f'{eval_path}/outputs/{to_filename(nix,0)}',mesh)
                save_instance(y,f'{eval_path}/outputs/{to_filename(nix,dt)}',mesh)

        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float32):
            rms = compute_errors(y,yt,mesh)
        sofar.append(rms["129_z_500"])
        print(f'{get_date_str(nix)}, {i}/{len(dataloader)}, z500 {rms["129_z_500"]}', "ohp %.2f pm %.2f"%(np.sqrt(np.mean(np.array(sofar)**2)), np.std(sofar)/(len(sofar)**0.5)))
        with open(f'{eval_path}/errors/{get_date_str(nix)}+{dt}.pickle','wb') as f:
            pickle.dump(rms,f)

def EvalGFS(eval_path):
    #eval_path = '/fast/evaluation/Neoreal_335M/'
    with open(f'{eval_path}/model.pickle','rb') as f:
        model = pickle.load(f)

    os.makedirs(f'{eval_path}/outputs/',exist_ok=True)
    os.makedirs(f'{eval_path}/errors/',exist_ok=True)

    device = torch.device('cuda:0')
    mesh = model.config.mesh
    mesh.config.source = 'era5' # this is only here becasue the version that was pickled didn't have this
    output_mesh = copy.deepcopy(mesh)
    dataset_config = mesh.config
    dataset_config.source = 'gfs'
    dataset_config.update()
    dataset = NeoWeatherDataset(NeoDataConfig(mesh=mesh,
                        #output_mesh=output_mesh,
                        timesteps=[dt],#model.config.timesteps,
                        requested_dates = get_dates((D(2020, 1, 1),D(2023,12,31))),
                        use_mmap = True,
                        only_at_z = [0,12],
                        clamp_output = np.inf,
                        ))
    dataset.check_for_dates()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=default_collate)

    _,stds,means = load_state_norm(mesh.wh_lev,dataset_config,with_means=True)


    model = model.to(device)
    sofar = []
    for i,sample in enumerate(dataloader):
    #for i,sample in enumerate([default_collate([dataset[0]])]):
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            x = sample[0][0].to(device) 
            dt = int(sample[1][1] / 3600) 
            #print("dt is", dt)
            nix = int(sample[0][1])
            y = model(x,dt=dt)
            x = x
            x,y = unnorm_output(x,y,model,dt)
            yt = unnorm(sample[1][0].to(device),mesh)
            if i % 50 == 0:
                print(f'Saving, x.shape {x.shape}, y.shape {y.shape}')
                save_instance(x,f'{eval_path}/outputs/{to_filename(nix,0,tags=["gfs"])}',mesh)
                save_instance(y,f'{eval_path}/outputs/{to_filename(nix,dt,tags=["gfs"])}',mesh)

        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float32):
            rms = compute_errors(y,yt,mesh)
            persistence = compute_errors(x,yt,mesh)
        sofar.append(rms["129_z_500"])

        cmptag = get_compare_tag(dataset.config)    

        print(f'{get_date_str(nix)}, {i}/{len(dataloader)}, {cmptag}, z500 {rms["129_z_500"]:.2f} (persistence {persistence["129_z_500"]:.2f})')

        with open(f'{eval_path}/errors/{to_filename(nix,dt,tags=[cmptag])}.pickle','wb') as f:
            pickle.dump(rms,f)

    exit()

def EvalChain(eval_path):
    #eval_path = '/fast/evaluation/Neoreal_335M/'
    with open(f'{eval_path}/model.pickle','rb') as f:
        model = pickle.load(f)

    os.makedirs(f'{eval_path}/outputs/',exist_ok=True)
    os.makedirs(f'{eval_path}/errors/',exist_ok=True)

    device = torch.device('cuda:0')
    mesh = model.config.mesh
    mesh.config.source = 'era5' # this is only here becasue the version that was pickled didn't have this
    dataset_config = mesh.config
    dataset = NeoWeatherDataset(NeoDataConfig(mesh=mesh, 
                        timesteps=[24,48,72,96,120,144,168,192,216,240],
                        requested_dates = get_dates((D(2020, 1, 1),D(2020,12,31))),
                        use_mmap = True,
                        only_at_z = [0,12],
                        clamp_output = np.inf,
                        ))
    dataset.check_for_dates()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=default_collate)

    _,stds,means = load_state_norm(mesh.wh_lev,dataset_config,with_means=True)


    model = model.to(device)
    sofar = []
    for i,sample in enumerate(dataloader):
    #for i,sample in enumerate([default_collate([dataset[0]])]):
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            x = sample[0][0].to(device) 
            for ss in sample[1:]:
                dt = int(ss[1] / 3600) 
                #print("dt is", dt)
                nix = int(sample[0][1])
                y = model(x,dt=24)
                _,xx = unnorm_output_partial(x,y,model,24)
                x,y = unnorm_output(x,y,model,24)
                yt = unnorm(ss[0].to(device),mesh)
                yyt = ss[0]
                xx = xx
                x = dataset.extend_and_clamp_input(xx.detach().to('cpu')[0],get_date(nix+dt*3600))
                x = x.reshape(1,*x.shape).to(device)

                with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float32):
                    rms = compute_errors(y,yt,mesh)
                print(f'{get_date_str(nix)}, {dt}, {rms["129_z_500"]:.2f}')

        #print(f'{get_date_str(nix)}, {i}/{len(dataloader)}, z500 {rms["129_z_500"]}', "ohp %.2f pm "%(np.sqrt(np.mean(np.array(sofar)**2)), np.std(sofar)/(len(sofar)**0.5)))
        #with open(f'{eval_path}/errors/{get_date_str(nix)}+{dt}.pickle','wb') as f:
        #    pickle.dump(rms,f)


def EvalChain2020(eval_path, input_dt=24, target_dt=24, dates=None):
  with torch.no_grad():
    #eval_path = '/fast/evaluation/Neoreal_335M/'
    with open(f'{eval_path}/model.pickle','rb') as f:
        model = pickle.load(f)

    os.makedirs(f'{eval_path}/outputs/',exist_ok=True)
    os.makedirs(f'{eval_path}/errors/',exist_ok=True)

    device = torch.device('cuda:0')
    mesh = model.config.mesh
    #print("uh mesh is", mesh, mesh.__dict__)
    if dates is None:
        dates = get_dates((D(2020, 1, 1),D(2020,12,31)))
    print("date range", dates[0], dates[-1])
    #dataset_config = mesh.config
    print("mesh", model.config.mesh.source)
    print("output mesh?", model.config.output_mesh.source)
    output_mesh = model.config.output_mesh
    dataset = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[output_mesh],
                        timesteps=[target_dt],#model.config.timesteps,
                        requested_dates = dates,
                        use_mmap = True,
                        only_at_z = [0,12],
                        clamp_output = np.inf,
                        ))
    #print("og output", model.config.output_mesh.config.source)
    #print("hey dataset", "mesh", dataset.config.mesh.config.source, "output_mesh", dataset.config.output_mesh.config.source)
    dataset.check_for_dates()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=default_collate)


    _,stds,means = load_state_norm(mesh.wh_lev, mesh, with_means=True)


    model = model.to(device)
    sofar = []
    for i,sample in enumerate(dataloader):
    #for i,sample in enumerate([default_collate([dataset[0]])]):
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            x = sample[0]
            x = [xx.to(device) for xx in x]
            #x = sample[0][0].to(device) 
            nsteps = target_dt//input_dt
            ss = sample[1]
            dt = sample_dts(sample)[0]#int(ss[1] / 3600) 
            #print(sample)
            #print("hey", target_dt, dt, sample_dts(sample))
            #exit()
            assert len(sample) == 2
            assert dt == target_dt
            #print("uh dt is", dt, len(sample))
            for step in range(nsteps):
                #print("dt is", dt)
                nix = int(sample[0][1])
                y = model(x,dt=input_dt)
                _,xx = unnorm_output_partial(x[0],y,model,input_dt)
                _,y = unnorm_output(x[0],y,model,input_dt)
                #xx = xx
                #x = dataset.extend_and_clamp_input(xx.detach().to('cpu')[0],get_date(nix+(step+1)*input_dt*3600))
                #x = x.reshape(1,*x.shape).to(device)
                x = [xx, x[1] + input_dt*3600]

            yt = unnorm(ss[0].to(device),mesh)
            with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float32):
                rms = compute_errors(y,yt,mesh)
            sofar.append(rms["129_z_500"])
            print(f'{get_date_str(nix)}, {dt}, {rms["129_z_500"]:.2f} sofar %.2f %.2f' % (np.sqrt(np.mean(np.array(sofar)**2)), np.std(sofar)/(len(sofar)**0.5)))

        #print(f'{get_date_str(nix)}, {i}/{len(dataloader)}, z500 {rms["129_z_500"]}', "ohp %.2f pm "%(np.sqrt(np.mean(np.array(sofar)**2)), np.std(sofar)/(len(sofar)**0.5)))
        #with open(f'{eval_path}/errors/{get_date_str(nix)}+{dt}.pickle','wb') as f:
        #    pickle.dump(rms,f)

def EvalAdapterOnly(eval_path, dates):
    os.makedirs(f'{eval_path}/outputs/',exist_ok=True)
    os.makedirs(f'{eval_path}/errors/',exist_ok=True)

    with open(f'{eval_path}/model.pickle','rb') as f:
        model = pickle.load(f)

    mesh1 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='gfs-28')
    mesh3 = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh1],
                                           outputs = [mesh3],
                                           timesteps=[0],
                                           requested_dates = dates,
                                           use_mmap = True,
                                           clamp_output=np.inf,
                                           ))
    data.check_for_dates()
    model = model.to('cuda')
    eval_loop(model, data, eval_path)

def eval_loop(model, data, eval_path):
    sofar = []
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=2, collate_fn=default_collate)
    mesh = data.config.outputs[0]
    for i,sample in enumerate(dataloader):
    #for i,sample in enumerate([default_collate([dataset[0]])]):
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            x = sample[0]
            x = [xx.to('cuda') for xx in x]
            dt = sample_dts(sample)[0]
            #print("dt is", dt)
            nix = int(sample[0][1])
            y = model(x,dt=dt)
            x,y = unnorm_output(x[0],y,model,dt)
            yt = unnorm(sample[1][0].to('cuda'),mesh)
            if i % 50 == 0:
                print(f'Saving, x.shape {x.shape}, y.shape {y.shape}')
                save_instance(x,f'{eval_path}/outputs/{to_filename(nix,0)}',mesh)
                save_instance(y,f'{eval_path}/outputs/{to_filename(nix,dt)}',mesh)

        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float32):
            rms = compute_errors(y,yt,mesh)
        sofar.append(rms["129_z_500"])
        print(f'{get_date_str(nix)}, {i}/{len(dataloader)}, z500 {rms["129_z_500"]}', "ohp %.2f pm %.2f"%(np.sqrt(np.mean(np.array(sofar)**2)), np.std(sofar)/(len(sofar)**0.5)))
        with open(f'{eval_path}/errors/{get_date_str(nix)}+{dt}.pickle','wb') as f:
            pickle.dump(rms,f)


ADAPTER_EVAL_DATES = get_dates((D(2022, 5, 1), D(2023, 8, 31)))


if __name__ == '__main__':
    #Eval2020('/fast/evaluation/Neoreal_335M/')
    #Eval2020('/fast/evaluation/Tardis_289M/')
    #Eval2020('/fast/evaluation/TardisL2FT_289M/')
    #  Eval2020('/fast/evaluation/TardisNeoL2FT_289M/')
    #Eval2020('/fast/evaluation/EnsSmall_166M/')
    #Eval2020('/fast/evaluation/SmolMenace_311M/')
    #Eval2020('/fast/evaluation/Tiny_311M/')
    #Eval2020('/fast/evaluation/Tiny2_311M/')
    #Eval2020('/fast/evaluation/neoquadripede_331M/')
    #Eval2020('/fast/evaluation/singlemenace_307M/')
    #Eval2020('/fast/evaluation/casio_266M/', dt=6)
    #Eval2020('/fast/evaluation/neocasio_341M/', dt=6)
    #EvalChain2020('/fast/evaluation/neoquadripede_72x24_331M/', input_dt=24, target_dt=72)
    #EvalChain2020('/fast/evaluation/neoquadripede_168x24_331M/', input_dt=24, target_dt=168)
    #EvalChain2020('/fast/evaluation/neoquadripede_120x24_331M/', input_dt=24, target_dt=120)
    #EvalChain2020('/fast/evaluation/neoquadripede_48x24_331M/', input_dt=24, target_dt=48)
    #EvalChain2020('/fast/evaluation/neoquadripede_96x24_331M/', input_dt=24, target_dt=96)
    #EvalChain2020('/fast/evaluation/neoquadripede48_331M/', input_dt=24, target_dt=48)
    #EvalChain2020('/fast/evaluation/neoquadripede48x2_331M/', input_dt=24, target_dt=96)
    #EvalChain2020('/fast/evaluation/singlemenace_120x24_307M/', input_dt=24, target_dt=120)
    #Eval2020('/fast/evaluation/TinyOper2_311M/')
    #Eval2020('/fast/evaluation/NeoEnc_244M/')
    #Eval2020('/fast/evaluation/Quadripede_218M/')
    #Eval2020('/fast/evaluation/neoTardisNeoL2FT_289M/')
    #Eval2020('/fast/evaluation/neoTardis22anl_289M/', dates=get_dates((D(2022, 8, 1),D(2022,12,30))))
    #Eval2020('/fast/evaluation/TinyOper22fct_311M/', dates=get_dates((D(2022, 8, 1),D(2022,12,30))))
    #Eval2020('/fast/evaluation/TinyOper2_22fct_311M/', dates=get_dates((D(2022, 8, 1),D(2022,12,30))))
    #Eval2022adapter('/fast/evaluation/neoTardis22fct_luigi2_289M/', dates=get_dates((D(2022, 8, 1),D(2022,12,30))))
    Eval2022adapter('/fast/evaluation/neoquadripede_neohegel_331M/', dates=get_dates((D(2022, 8, 1),D(2022,12,30))))
    #Eval2020('/fast/evaluation/SingleDec_240M/')
    #EvalChain2020('/fast/evaluation/TardisNeoL2FT_72x12_289M/', input_dt=12, target_dt=72)

    #EvalAdapterOnly('/fast/evaluation/WarioJAW_5M/', ADAPTER_EVAL_DATES)


    if 0:
        with open(f'/fast/evaluation/TardisL2FT_289M/model.pickle','rb') as f:
            model = pickle.load(f)
            pass
        



