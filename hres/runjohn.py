import sys
import time
from runs.launch import *
from hres.train import *
from hres.data import *
from torch.utils.data import default_collate
import torch
torch.manual_seed(0)
from matplotlib.colors import BoundaryNorm
import pytz


import numpy as np
np.random.seed(0)

stations_latlon = pickle.load(open('/fast/proc/hres_consolidated/consolidated/stations_latlon.pickle', 'rb'))

@launch(ddp=0, log=False)
def john():
    dataset = HresDataset(batch_size=512)
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True)

    conf.nope = True
    conf.optim = 'adam'
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


# load normalization
with open("/fast/consts/normalization.pickle", "rb") as f:
    normalization = pickle.load(f)

neocache = {}
def interpget(src, toy, hr, a=300, b=400):
    global neocache
    if src not in neocache:
        neocache[src] = {}

    def load(xx, hr):
        if (xx,hr) in neocache[src]:
            return neocache[src][(xx, hr)]
        #print("loading", src, xx, hr)
        if conf.HALF:
            f = torch.HalfTensor
        else:
            f = torch.FloatTensor
        ohp = f(((np.load("/fast/consts/"+'/%s/%d_%d.npy' % (src, xx, hr)) - a) / b))
        neocache[src][(xx,hr)] = ohp
        return ohp
    avg = load(toy, hr)
    return avg


def build_input(model,era5,interp,idx,statics,date,primary_compute='cuda'):
    assert era5["sfc"].shape[0] == 1
    if model.do_pressure:
        pr = era5["pr"].to(primary_compute)
        pr_sample = pr[0, idx[..., 0], idx[..., 1], :, :]
        pr_sample = torch.sum(pr_sample * interp[:,:,:,:,None,None], axis=3)[0]

    if model.do_radiation:
        soy = date.replace(month=1, day=1)
        toy = int((date - soy).total_seconds()/86400)
        if toy % 3 != 0:
            toy -= toy % 3
        rad = interpget("neoradiation_1", toy, date.hour)
        ang = interpget("solarangle_1", toy, date.hour, a=0, b=180/np.pi)
        sa = torch.sin(ang)
        ca = torch.cos(ang)
        arrs = [rad, sa, ca]

        extra = []
        for a in arrs:
            a = a.cuda()
            exa = a[idx[..., 0], idx[..., 1]]
            exa = torch.sum(exa * interp, axis=3)[0]
            del a
            extra.append(exa[:,:,None])

    else:
        extra = []


    sfc = era5["sfc"].to(primary_compute)
    sfc_sample = sfc[0, idx[..., 0], idx[..., 1], :]
    sfc_sample = torch.sum(sfc_sample * interp[:,:,:,:,None], axis=3)[0]
    sera5 = statics["era5"]
    """
    sera5[:, :, 0] /= 1000. * 9.8
    sera5[:, :, 1] /= 7.
    sera5[:, :, 3] /= 20.
    sera5[:, :, 4] /= 20.
    """
    sfc_sample = torch.cat([sfc_sample, sera5] + extra, axis=-1)

    if model.do_pressure:
        center_pr = pr_sample[:,0]
        pr_sample = pr_sample[:, 1:]



    center_sfc = sfc_sample[:,0]

    sfc_sample = sfc_sample[:, 1:]
    sq = int(np.sqrt(sfc_sample.shape[1]))
    sfc_sample = sfc_sample.permute(0, 2, 1).view(-1, sfc_sample.shape[2], sq, sq)

    static_keys = ["mn30", "mn75"]
    modis_keys = ["modis_"+x for x in static_keys]
    static = {x: statics[x] for x in static_keys + model.do_modis*modis_keys}
    center = {x: static[x][:,0,0] for x in static_keys}
    if model.do_modis:
        for x in modis_keys:
            center[x] = torch.nn.functional.one_hot(static[x][:,0].long(), 17)
    for x in static_keys:
        sq = int(np.sqrt(static[x].shape[1]-1))
        static[x] = static[x][:,1:].view(-1, sq, sq, 3)
        if x.startswith("mn"):
            #print("hullo", static[x][:,:,:,0].mean(), center[x][:,None,None].mean(), static[x].shape, center[x].shape)
            #static[x][:,:,:,0] = (static[x][:,:,:,0] - center[x][:,None,None])*(1./150)
            #static[x][:,:,:,1:] /= 20.
            #center[x] = center[x]*(1./1000)
            #static[x][:,:,:,0] = (static[x][:,:,:,0] - center[x][:,None,None])*(1./150)
            #static[x][:,:,:,1:] /= 20.
            #center[x] = center[x]*(1./1000)
            pass
        if model.do_modis:
            #modis = static["modis_"+x][:, 1:].view(-1, sq, sq, 17)
            modis = static["modis_"+x][:, 1:].view(-1, sq, sq)#, 17)
            modis = torch.nn.functional.one_hot(modis.long(), 17)
            static[x] = torch.cat((static[x], modis), dim=3)
        static[x] = static[x].permute(0, 3, 1, 2)
    inp = {}

    if model.do_pressure:
        sq = int(np.sqrt(pr_sample.shape[1]))
        pr_sample = pr_sample.view(-1, sq, sq, 5, 28).permute(0, 3, 4, 1, 2)
        inp["pr"] = pr_sample
    #print("pr shape", inp["pr"].shape, inp["sfc"].shape)
    inp["sfc"] = sfc_sample
    for k in static_keys:
        inp[k] = static[k]
    inp["center"] = torch.cat([center[x][:, None] if len(center[x].shape)==1 else center[x] for x in static_keys + model.do_modis * modis_keys], dim=1)
    print("hey", inp["center"].shape, center_sfc.shape)
    inp["center"] = torch.cat([inp["center"], center_sfc], dim=1)
    inp = {x: y.half() for x, y in inp.items()}
    return inp


#if 1:


vname = 'hawaii'
vname = 'oahu'
vname = 'bay'

if vname == 'hawaii':
    vshape = 155, 246 #hawaii
elif vname == 'bay':
    vshape = 73, 106 #bayarea
elif vname == 'oahu':
    vshape = 126, 181 #oahu, 0.5km resolution



with torch.no_grad():
    interps, idxs = pickle.load(open(f"/fast/ignored/hres/{vname}_interps.pickle", "rb"))
    interps = torch.tensor(interps).unsqueeze(0).to('cuda')
    idxs = torch.tensor(idxs).unsqueeze(0).to('cuda')
    statics = pickle.load(open(f"/fast/ignored/hres/{vname}_statics.pickle", "rb"))
    statics = {x: torch.tensor(y).to('cuda') for x, y in statics.items()}

    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    """
    s = statics["mn75"]
    s = s[:, 0, 0].view(73, 106).cpu().numpy()
    print(s.shape);
    plt.imshow(s, origin='lower', extent=ex)
    plt.savefig('hres/mn75.png'); exit()
    """



    dataset = HresDataset(batch_size=None)
    #print("huh", datetime(1970,1,1)+timedelta(seconds=int(dataset[20084][-1]))); exit()
    for time_i in range(48):
        date = datetime(2019, 3, 10, 1, 0, 0) + timedelta(hours=time_i)
        #date = datetime(2019, 7, 3, 0, 0, 0) + timedelta(hours=time_i)
        date = datetime(1994, 10, 1, 0, 0, 0) + timedelta(hours=time_i)
        #era5 = HresDataset.load_era5(date)
        era5, data, stations, is_valid, weights,  _ = dataset.get_by_date(date, do_era5=True)
        era5 = {x: torch.tensor(y).unsqueeze(0).to('cuda') for x, y in era5.items()}
        print("date is", date)
        #model = HresModel(do_modis=False)

        bayarea = pickle.load(open(f"/fast/ignored/hres/{vname}_pts.pickle", "rb"))
        #print(bayarea.shape); exit()

        #vshape = 73, 106 #bayarea
        #vshape = 155, 246 #hawaii

        crs = ccrs.PlateCarree()
        ex = [np.min(bayarea[:,:,1]), np.max(bayarea[:,:,1]), np.min(bayarea[:,:,0]), np.max(bayarea[:,:,0])]


        model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True)
        save_path = '/fast/ignored/runs_hres/run_Apr22-doot-bugfix_20240422-132101/model_keep_step298400_loss0.067.pt'
        save_path = '/fast/ignored/runs_hres/run_Apr25-wallstick_20240425-104123/model_keep_step97000_loss0.049.pt'
        save_path = '/fast/ignored/runs_hres/run_Apr25-wallstick_20240425-104123/model_keep_step178000_loss0.044.pt'
        save_path = '/fast/ignored/runs_hres/run_Apr25-wallstick_20240425-104123/model_keep_step227500_loss0.041.pt'
        save_path = '/fast/ignored/runs_hres/run_Apr25-wallstick_20240425-104123/model_keep_step283500_loss0.039.pt'
        save_path = '/fast/ignored/runs_hres/run_Apr25-wallstick_20240425-104123/model_keep_step309000_loss0.039.pt'
        save_path = '/fast/ignored/runs_hres/run_May5-multivar_20240508-103121/model_step56000_loss0.158.pt'
        save_path = '/fast/ignored/runs_hres/run_May5-multivar_20240508-103121/KEEPMEmodel_step61000_loss0.160.pt'
        save_path = '/fast/ignored/runs_hres/run_May5-multivar_20240510-144336/model_keep_step133500_loss0.144.pt'
        checkpoint = torch.load(save_path,map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to('cuda')
        model.eval()
        t0 = time.time()
        full = np.arange(idxs.shape[1])
        print(len(full))
        outs = [] ; bils = []
        for chunk in np.array_split(full, 10):
            interpsx = interps[:,chunk]
            idxsx = idxs[:,chunk]
            staticsx = {x: y[chunk] for x, y in statics.items()}
            inpx = build_input(model, era5, interpsx, idxsx, staticsx, date)
            #bils.append(inpx["center"][:,-4+2])
            bils.append(inpx["center"][:,-12:-8])
            print(bils[0].shape)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(inpx)
            outs.append(out.cpu().detach().numpy())
        out = np.concatenate(outs, axis=0)
        out = out.reshape(*vshape, -1)
        bil = torch.concatenate(bils, axis=0).view(*vshape,4).cpu().numpy()
        print(bil.shape)
        #out = out.reshape(73, 106, -1).cpu().detach().numpy()
        print("end", time.time()-t0)

        # initialize cartopy map 

        #self.normalization['165_10u']

        def n(arr, norm):
            arr = arr.astype(np.float32)
            return arr * np.sqrt(normalization[norm][1][0]) + normalization[norm][0][0]

        tmp = n(out[:,:,0], '167_2t')
        dpt = n(out[:,:,1], '168_2d')
        mslp = n(out[:, :, 2], '151_msl')
        ucomp = n(out[:,:,3], '165_10u') * 1.94
        vcomp = n(out[:,:,4], '166_10v') * 1.94
        tmp_bil = n(bil[...,2], '167_2t')
        ucomp_bil = n(bil[...,0], '165_10u') * 1.94
        vcomp_bil = n(bil[...,1], '166_10v') * 1.94

        highres = 1
        if 0:
            highres = 0
            tmp = n(inp["center"][:,-4+2].view(*vshape).cpu().numpy(), '167_2t')
            dpt = tmp * 0
            mslp = n(inp["center"][:,-4+3].view(*vshape).cpu().numpy(), '151_msl')
            ucomp = n(inp["center"][:,-4+0].view(*vshape).cpu().numpy(), '165_10u')
            vcomp = n(inp["center"][:,-4+1].view(*vshape).cpu().numpy(), '166_10v')
            extra = "bil"
        else:
            extra = ""
        
        s = lambda x: x * np.sqrt(normalization['167_2t'][1][0])
        su = lambda x: x * np.sqrt(normalization['165_10u'][1][0])
        sv = lambda x: x * np.sqrt(normalization['166_10v'][1][0])

        min_lat, max_lat = bayarea[:,:,0].min(), bayarea[:,:,0].max()
        min_lon, max_lon = bayarea[:,:,1].min(), bayarea[:,:,1].max()
        stlat = []
        stlon = []
        errs1 = []
        errs2 = []
        werrs1 = []
        werrs2 = []
        #done = []
        for idxst, pt in enumerate(stations):
            lat, lon = stations_latlon[pt]
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                stlat.append(lat)
                stlon.append(lon)
                obs = data[idxst].cpu().numpy()
                dist = np.abs((lat - bayarea[:,:,0])**2 + (lon - bayarea[:,:,1])**2)
                closest = np.unravel_index(np.argmin(dist), dist.shape)
                mod = out[closest]
                #bil = inp["center"][:, -4+2].view(*vshape)[closest].cpu().numpy()
                if np.isnan(obs[0]): continue
                err_tmp1 = s(mod[0] - obs[0])
                err_tmp2 = s(bil[closest][2] - obs[0])
                print(lat, lon, "hres", err_tmp1, "bil", err_tmp2, obs)
                errs1.append(err_tmp1)
                errs2.append(err_tmp2)
                if np.isnan(obs[3]): continue
                bilu = bil[closest][0]
                bilv = bil[closest][1]
                werrs1.append(su(mod[3] - obs[3]))
                werrs1.append(sv(mod[4] - obs[4]))
                werrs2.append(su(bilu - obs[4]))
                werrs2.append(sv(bilv - obs[3]))
        rms = lambda x: np.sqrt(np.mean(np.square(x)))
        print("temp hres", rms(errs1), "bilinear", rms(errs2))
        print("winds hres", rms(werrs1), "bilinear", rms(werrs2))


        #ucomp *= 1.94
        #vcomp *= 1.94

        if 1: # Joan's 2x2 plot

            ax = plt.subplot(2,2,1, projection=crs)
            ax.set_extent(ex, crs=crs)
            ax.coastlines(resolution='10m')
            ax.set_title('temp')

            #wspd = np.sqrt(out[:,:,3]**2 + out[:,:,4]**2)
            #wdir = np.arctan2(out[:,:,3], out[:,:,4])
            ax.scatter(stlon, stlat, transform=ccrs.PlateCarree(), s=1, color='red')
            a = ax.imshow(tmp, origin='lower', interpolation='bilinear', extent=ex)
            cb = plt.colorbar(a,fraction=0.031, pad=0.04)
            cb.ax.tick_params(labelsize=8)

            ax = plt.subplot(2,2,2, projection=crs)
            ax.set_extent(ex, crs=crs)
            ax.coastlines(resolution='10m')
            ax.set_title('dewpoint')
            #ax.imshow(bil, origin='lower', interpolation='bilinear', extent=ex)
            a = ax.imshow(dpt, origin='lower', interpolation='bilinear', extent=ex)
            cb = plt.colorbar(a,fraction=0.031, pad=0.04)
            cb.ax.tick_params(labelsize=8)


            ax = plt.subplot(2,2,3, projection=crs)
            ax.set_extent(ex, crs=crs)
            ax.coastlines(resolution='10m')
            ax.set_title('mslp')
            a = ax.imshow(mslp, origin='lower', interpolation='bilinear', extent=ex)
            cb = plt.colorbar(a,fraction=0.031, pad=0.04)
            cb.ax.tick_params(labelsize=8)

            ax = plt.subplot(2,2,4, projection=crs)
            ax.set_extent(ex, crs=crs)
            ax.coastlines(resolution='10m')
            ax.set_title('10m winds')
            # plot wind speed in the background
            wspd = np.sqrt(ucomp**2 + vcomp**2)
            a = ax.imshow(wspd, origin='lower', interpolation='bilinear', extent=ex)
            cb = plt.colorbar(a,fraction=0.031, pad=0.04)
            cb.ax.tick_params(labelsize=8)
            # plot wind barbs with ucomp and vcomp for the u and v components of wind
            ax.barbs(bayarea[:,:,1], bayarea[:,:,0], ucomp, vcomp, regrid_shape=15, length=3., sizes=dict(emptybarb=0.3))#, length=6)#, regrid_shape=20)
            print("means", out[:,:,0].mean(), bil.mean())
            plt.tight_layout()
            plt.savefig('hres/ohp%s.png'%extra, bbox_inches='tight', dpi=300)
            exit()

        if 1:
            row = 0
            plt.figure(figsize=(12,8))
            for ucompx,vcompx,tmpx in [(ucomp,vcomp,tmp), (ucomp_bil,vcomp_bil,tmp_bil)]:
                print("row",row)
                ax = plt.subplot(2,2,1+row*2, projection=crs)
                ax.set_extent(ex, crs=crs)
                ax.coastlines(resolution='10m')
                ax.set_title(f'{date.replace(tzinfo=pytz.utc).astimezone(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d %-I %p")} PT |   Surface Winds (kts)')
                # plot wind speed in the background
                wspd = np.sqrt(ucompx**2 + vcompx**2)
                wspd = wspd 

                boundaries = np.linspace(0, 20, num=21)  # Adjust num for more or fewer intervals
                norm = BoundaryNorm(boundaries, ncolors=256)
                a = ax.imshow(wspd, origin='lower', interpolation='bilinear', extent=ex,norm=norm)
                cb = plt.colorbar(a,fraction=0.031, pad=0.04)
                cb.ax.tick_params(labelsize=8)
                # plot wind barbs with ucomp and vcomp for the u and v components of wind
                ax.barbs(bayarea[:,:,1], bayarea[:,:,0], ucompx, vcompx, regrid_shape=15, length=3., sizes=dict(emptybarb=0.3))#, length=6)#, regrid_shape=20)

                ax = plt.subplot(2,2,2+row*2, projection=crs)
                ax.set_extent(ex, crs=crs)
                ax.coastlines(resolution='10m')
                ax.set_title('Surface Temperature (F)')

                #wspd = np.sqrt(out[:,:,3]**2 + out[:,:,4]**2)
                #wdir = np.arctan2(out[:,:,3], out[:,:,4])
                tmp = (tmpx - 273.15) * 9/5 + 32
                boundaries = np.linspace(32, 80, num=13)  # Adjust num for more or fewer intervals
                norm = BoundaryNorm(boundaries, ncolors=256)
                a = ax.imshow(tmp, origin='lower', interpolation='bilinear', norm=norm,extent=ex)

                cb = plt.colorbar(a,fraction=0.031, pad=0.04)
                cb.ax.tick_params(labelsize=8)
                row+=1

            vispath = f'/fast/to_srv/hres/{vname}May14'
            os.makedirs(vispath, exist_ok=True)
            plt.subplots_adjust(wspace=0.11)
            plt.savefig(f'{vispath}/date={date.strftime("%Y%m%d%H")},_neovis.png', bbox_inches='tight', dpi=150)
            import neovis
            gif = neovis.Neovis(vispath)
            gif.make()
            plt.clf()
            plt.close()


exit()
if __name__ == '__main__':
    run(locals().values())
