from utils import *
from train import *

ROLLOUT = True
ROLLOUT = 0

outpath = "/fast/validation/val1113"

args = parse_args()
args.resume = "_"+args.activity+"_"

mesh = meshes.LatLonGrid(subsamp=1, levels=levels_medium)
#model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=1024, depth=10, drop_path=0)
#model.output_deltas = True
#model.do_sub = False
#model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, drop_path=0)
#model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=512, depth=72, drop_path=0, skip_every=None, window_size=(2,3,6))
model = ForecastStepSwin3D(mesh, patch_size=(4,8,8), conv_dim=768, depth=24, drop_path=0)
model.output_deltas = True
model.do_sub = False


w = WeatherTrainer(args,mesh,model)
w.preserved_conf.cpu_only = 1 
w.preserved_conf.clamp = 12
w.preserved_conf.clamp_output = np.inf
w.preserved_conf.DH = 24
w.preserved_conf.output_DH = 24
Y = 2018
#Y = 2018
#Y = 2007
Y = 2020
w.preserved_conf.dates = get_dates((D(Y, 1, 1), D(Y, 12,30)))
w.preserved_conf.only_at_z = [0,12]
w.preserved_conf.HALF = True
#w.preserved_conf.clamp = 12
w.setup_data()
w.setupTraining()
w.active_model.eval()

with torch.no_grad():
    z500s = []
    t500s = []
    tots = defaultdict(list)
    bydate = {}
    for i_batch, batch_ in tqdm(enumerate(w.data_loader),total=len(w.data_loader),desc="Epoch Progress",ascii=' >='):
        x,yt,dates = batch_
        for d in dates: print(datetime(1970,1,1)+timedelta(seconds=int(d)))
        B,N1,N2,D = x.shape 

        yt_gpu = yt.to(w.primary_compute)
        x_gpu = x.to(w.primary_compute)

        nL = mesh.n_levels
        nP = mesh.n_pr_vars
        nPL = mesh.n_pr; assert nPL == nPL
        nS = mesh.n_sfc_vars

        state_norm,matrix_std, matrix_mean = load_state_norm(mesh.wh_lev, with_means=True)
        print("uih", matrix_std.shape, matrix_mean.shape)

        matrix_pr_std = matrix_std[:nPL]
        matrix_pr_std.shape = (nP,nL)
        matrix_pr_mean = matrix_mean[:nPL]
        matrix_pr_mean.shape = (nP,nL)
        matrix_sfc_mean = matrix_mean[nPL:]
        matrix_sfc_std = matrix_std[nPL:]
        if ROLLOUT:
            with torch.autocast(enabled=bool(w.conf.HALF), device_type='cuda', dtype=torch.float16):
                N = 5
                for i in range(N):
                    dh = (i+1)*w.conf.DH
                    y_gpu = w.active_model(x_gpu)
                    y_gpu = x_gpu[...,:y_gpu.shape[-1]] + y_gpu*w.delta_norm_matrix_gpu[...,:y_gpu.shape[-1]]
                    cpu = y_gpu.cpu().numpy().astype(np.float32)
                    xpr = cpu[...,:nPL]
                    #print("xpr", xpr.shape, matrix_pr_std.shape, matrix_pr_mean.shape)
                    xpr.shape = (xpr.shape[1], xpr.shape[2], nP, nL)
                    xpr = matrix_pr_mean + matrix_pr_std * xpr
                    xsfc = cpu[...,nPL:]
                    xsfc = xsfc[0]
                    xsfc = matrix_sfc_mean + matrix_sfc_std * xsfc
                    dic = {}
                    for i, k in enumerate(cloud_pressure_vars):
                        dic[k] = xpr[:,:,i,:]
                    for i, k in enumerate(cloud_sfc_vars):
                        dic[k] = xsfc[:,:,i]
                    dic["levels"] = mesh.levels
                    np.savez(outpath+"/"+str(int(d))+"_"+str(dh), **dic)

                    d = dates[0] + 3600 *(i+1)*w.conf.DH
                    extra = w.weather_data.extra_variables(d)[np.newaxis].to(w.primary_compute)
                    x_gpu = torch.cat((y_gpu, extra), axis=-1)


        else:
            with torch.autocast(enabled=bool(w.conf.HALF), device_type='cuda', dtype=torch.float16):
                y_gpu = w.active_model(x_gpu)

            #target_gpu = (yt_gpu - x_gpu[...,:y_gpu.shape[-1]])/w.delta_norm_matrix_gpu[...,:y_gpu.shape[-1]]
            #y_gpu = x_gpu[...,:y_gpu.shape[-1]] + y_gpu*self.delta_norm_matrix_gpu[...,:y_gpu.shape[-1]]

            if w.conf.HALF: yt_gpu = yt_gpu.half()

            print("computing errors")
            d = datetime(1970,1,1)+timedelta(seconds=int(d))
            bydate[d] = {}
            rms, extra = w.computeErrors(x_gpu,y_gpu,yt_gpu, do_print=False)
            for i, v in enumerate(pressure_vars + sfc_vars):
                tots[v].append(rms[i])
                bydate[d][v] = rms[i]
            for v in extra:
                if "_500" in v:
                    tots[v].append(extra[v])
                bydate[d][v] = extra[v]
            with open("bydate.pickle", "wb") as f:
                pickle.dump(bydate, f)
            for v in sorted(tots.keys()):
                print(v, "neo: %.3f pm %.3f" % (np.sqrt(np.mean(np.array(tots[v])**2)), np.std(tots[v])))
                #if v == "129_z_500": print(tots[v])
        continue
        z500 = extra["129_z_500"]
        z500s.append(z500)
        print("z500", z500, "avg: %.2f pm %.2f" % (np.mean(z500s), np.std(z500s)))

        t500 = extra["130_t_500"]
        t500s.append(t500)
        print("t500", t500, "avg: %.2f pm %.2f" % (np.mean(t500s), np.std(t500s)))
