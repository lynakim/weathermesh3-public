
import os 
import json
import sys
sys.path.append('/fast/wbhaoxing/deep')
from utils import *
from evals import *
from evals.tc.tclib import *
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_forecast_hour(path):
    if "+" not in path:
        return 0
    return int(os.path.basename(path).split('.')[0].split('+')[-1])

def get_forecast_hour_new(path):
    assert "+" not in path, "+ is in path, doesn't look like new format to me!"
    return int(os.path.basename(path).split('.')[0])

def get_forecast_at(path):
    return date_str2date(os.path.basename(path).split('.')[0].split('+')[0])

def get_valid_at(path):
    return get_forecast_at(path) + timedelta(hours=get_forecast_hour(path))

def get_forecast_run(startpath):
    hash = os.path.basename(startpath).split('.')[1]
    forecast_at = os.path.basename(startpath).split('.')[0]

    fs = []
    for f in os.listdir(os.path.dirname(startpath)):
        if f.startswith(forecast_at) and hash in f:
            fs.append(os.path.join(os.path.dirname(startpath),f))
    fs.sort(key=lambda x: get_forecast_hour(x))
    return fs

def get_forecast_run_new(output_path, forecast_at):
    fs = []
    for f in os.listdir(f"{output_path}/{forecast_at}/det"):
        if f.endswith(".npy"):
            fs.append(f"{output_path}/{forecast_at}/det/{f}")
    fs.sort(key=lambda x: get_forecast_hour_new(x))
    return fs

def get_meta(file):
    hash = os.path.basename(file).split('.')[1]
    metapath = os.path.dirname(file)+f'/meta.{hash}.json'
    with open(metapath,'r') as f:
        js = json.load(f)
    return js

def get_track(startpath, appox_loc):
    pass

from scipy.ndimage import gaussian_filter
from neovis import Neovis

#def vorticity(u,v,mesh,ks=8):
#    nlat = 
#    pms = np.arange(-ks,ks+1)
#    for i 


def get_pred_loc(x, real_loc,mesh):
    pm = 5
    bbox = [real_loc[0]-pm, real_loc[1]-pm, real_loc[0]+pm, real_loc[1]+pm] 
    smesh = bbox_mesh(mesh,bbox)
    xs = select_bbox(x, mesh, bbox)
    amin = np.unravel_index(np.argmin(xs), xs.shape)
    loc = smesh.lats[amin[0]], smesh.lons[amin[1]]
    return loc

def plot_hur_fcst(name, hurname, output_path, forecast_at, rerun=True):
    outpath = f'/fast/to_srv/hur/{name}/{hurname}/'   
    bbox = get_hur_bbox(hurname)
    os.makedirs(outpath,exist_ok=True)
    if 0:
        for f in os.listdir(outpath):
            if f.endswith('.png'):
                os.remove(os.path.join(outpath,f))

    run = get_forecast_run_new(output_path, forecast_at)
    if len(run) == 0: 
        print(f'No files found for {start}')
        return
    res = SimpleNamespace()
    res.realpath = [] ; res.predpath = [] 
    official_track = get_hur_track(hurname)
    #official_tracks = get_hur_tracks(list(storms_2024.keys()))
    print(f"Official track for {hurname} starts at UTC {datetime.utcfromtimestamp(official_track[0][2]).strftime('%Y-%m-%d %H:%M:%S')} ends at {datetime.utcfromtimestamp(official_track[-1][2]).strftime('%Y-%m-%d %H:%M:%S')}")
    forcast_hours = [get_forecast_hour_new(f) for f in run]
    res.forcast_hours = forcast_hours
    if rerun or not os.path.exists(f'{outpath}/{forecast_at}_res.pkl'):
        for i, f in enumerate(run):
            with Timer('load_instance',print=True):
                xx, mesh = load_instance(f)
                xx = select_bbox(xx,mesh,bbox)
                mesh = bbox_mesh(mesh,bbox)
            #x = x[:, :, mesh.full_varlist.index('151_msl')]
            #var = '129_z_850'
            h = get_forecast_hour_new(f) if get_forecast_hour_new(f) is not None else 0
            valid_at_nix = to_unix(date_str2date(forecast_at) + timedelta(hours=int(h)))
            real_loc = get_hur_loc(official_track,valid_at_nix)
            loc = None
            #for var in ['129_z_850','131_u_850','132_v_850']:
            for var in ['129_z_850','129_z_500','129_z_850GAUS','131_u_850','132_v_850']:
                vvar = var
                if var.endswith('GAUS'):
                    vvar = var[:-4]
                x = xx[:, :, mesh.full_varlist.index(vvar)]
                if var.endswith('GAUS'):
                    x = gaussian_filter(x, sigma=2)
                if real_loc is not None:
                    loc = get_pred_loc(x,real_loc,mesh)
                    if var == '129_z_850GAUS':
                        res.realpath.append(real_loc) ; res.predpath.append(loc)
                print(f'{hurname} forecast_hour {h} | forecast_at {forecast_at} | valid_at {get_date_str(date_str2date(forecast_at) + timedelta(hours=int(h)))} | {var}')
                # Set up Cartopy map
                # plt.clf()
                # plt.close()
                # fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(7,4))
                # ax.set_extent([bbox[1], bbox[3], bbox[2], bbox[0]], crs=ccrs.PlateCarree())  # Setting map extent
                # ax.coastlines()
                # im = ax.imshow(x, extent=[bbox[1], bbox[3], bbox[2], bbox[0]], origin='upper', transform=ccrs.PlateCarree())#,vmin=12500,vmax=15000)
                # if real_loc is not None:
                #     #ax.plot(real_loc[1], real_loc[0], 'bo', markersize=2, transform=ccrs.PlateCarree())
                # if loc is not None:
                #     ax.plot(loc[1], loc[0], 'ro', markersize=2, transform=ccrs.PlateCarree())
                # ax.set_title(f'forecast_hour {h:03} | forecast_at {forecast_at} | valid_at {get_date_str(date_str2date(forecast_at) + timedelta(hours=int(h)))} | {var}',fontsize=10)
                # plt.colorbar(im,ax=ax,shrink=0.5,aspect=10)
                # fig.tight_layout()
                # print(f'saving {outpath}/fcst-hour={h:03},fcst-at={forecast_at},var={var},_neovis.png')
                # plt.savefig(f'{outpath}/fcst-hour={h:03},fcst-at={forecast_at},var={var},_neovis.png',dpi=100)
        dist = list(map(haver_dist,res.realpath,res.predpath))
        res.dist = dist
        with open(f'{outpath}/{forecast_at}_res.pkl','wb') as f:
            pickle.dump(res,f)
    else:   
        with open(f'{outpath}/{forecast_at}_res.pkl','rb') as f:
            res = pickle.load(f)
     
    # gif = Neovis(outpath)
    # gif.make()

    #nhc_pred = get_pred_track(hurname,forecast_at)
    cmp_predpath = get_gfs_pred_track(hurname,forecast_at)
    if cmp_predpath is not None:
        cmp_predpath = [get_hur_loc(cmp_predpath,to_unix(date_str2date(forecast_at) + timedelta(hours=int(h)))) for h in forcast_hours]
        cmp_predpath = [x for x in cmp_predpath if x is not None]
        cmp_dist = list(map(haver_dist,res.realpath[:len(cmp_predpath)],cmp_predpath))
        with open(f'{outpath}/{forecast_at}_cmp.pkl','wb') as f:
            cmp = SimpleNamespace()
            cmp.predpath = cmp_predpath
            cmp.dist = cmp_dist
            cmp.forcast_hours = forcast_hours
            pickle.dump(cmp,f)
    else:
        cmp_predpath = None
        cmp_dist = [0]*len(res.realpath)

    plt.clf()
    plt.close()
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#e7e7e7')  # Set the background color to match the ocean in A
    ax.set_facecolor('#e7e7e7')  # Set the axes background color
    dist = list(map(haver_dist,res.realpath,res.predpath))
    #plt.axvline((datetime(2022, 9, 28, 19, 10)-date_str2date(forecast_at)).total_seconds() / 3600, color='r', linestyle='--')
    plt.plot(forcast_hours[:len(dist)],dist,label='WB',color='#009410')  # Use WB color from A
    plt.plot(forcast_hours[:len(cmp_dist)],cmp_dist,label='Other',color='#0038FF')  # Use NWS color from A
    plt.grid(color='#919191',alpha=0.4)  # Set grid color to match borders in A
    plt.legend()
    plt.ylabel('Ground Track Distance Error (km)')
    plt.xlabel('Forecast Hour')
    plt.title(f'Track Error for {hurname}, forecasted at {forecast_at}')
    os.makedirs(f'{outpath}/error/',exist_ok=True)
    plt.savefig(f'{outpath}/error/fcst-at={forecast_at},_neovis.png', facecolor=fig.get_facecolor(),dpi=200)  # Ensure saved figure matches specified background color


    gif = Neovis(f'{outpath}/error')
    gif.make()

    plt.clf()
    plt.close()
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(7,4))
    ax.set_extent([bbox[1], bbox[3], bbox[2], bbox[0]], crs=ccrs.PlateCarree())  # Setting map extent
    ax.add_feature(cfeature.LAND, facecolor='#B5B5B5')
    ax.add_feature(cfeature.OCEAN, facecolor='#e7e7e7')
    cc = '#919191'
    lw1 = 0.4
    ax.add_feature(cfeature.BORDERS,linewidth=lw1,color=cc)
    ax.add_feature(cfeature.STATES,linewidth=lw1,edgecolor=cc)
    ax.add_feature(cfeature.COASTLINE,linewidth=lw1,color=cc)

    #plot_polygon(ax,hurname,forecast_at)
    lw=1;ms=3
    if cmp_predpath is not None:
        ax.plot([x[1] for x in cmp_predpath],[x[0] for x in cmp_predpath],label='NWS',color='#0038FF',linewidth=lw)
        ax.plot([x[1] for x in cmp_predpath],[x[0] for x in cmp_predpath],'.',color='#0038FF',markersize=ms)
    ax.plot([x[1] for x in res.predpath],[x[0] for x in res.predpath],label='WB',color='#009410',linewidth=lw)
    ax.plot([x[1] for x in res.predpath],[x[0] for x in res.predpath],'.',color='#009410',markersize=ms)
    ax.plot([x[1] for x in res.realpath],[x[0] for x in res.realpath],label='Actual',color='#000000',linewidth=lw)
    ax.plot([x[1] for x in res.realpath],[x[0] for x in res.realpath],'.',color='#000000',markersize=ms)

    ax.legend()
    ax.set_title(f'track for {hurname}, forecasted at {forecast_at}')
    os.makedirs(f'{outpath}/error_map/',exist_ok=True)
    plt.savefig(f'{outpath}/error_map/fcst-at={forecast_at},_neovis.png',dpi=200)

    gif = Neovis(f'{outpath}/error_map')
    gif.make()
    print(f'Done {hurname} forecasted at {forecast_at}')
    return dist, res.predpath, [date_str2date(forecast_at) + timedelta(hours=int(h)) for h in forcast_hours][:len(dist)]

def get_outpath(name):
    return f'/fast/to_srv/hur/{name}/'

def many_fcst(name,path,dates,rerun=True):
    if 'neocasioquad_673M' in path:
        name+="_era5"
    if 'yung' in path:
        name+="_undertrained"
    print(f'Running {name}')
    results = []
    outpath = get_outpath(name)
    hurname = name.split('_')[0]
    bbox = get_hur_bbox(hurname)
    if rerun or not os.path.exists(f'{outpath}/results.pkl'):
        for date in dates:
            date_str2date(date)
            r = plot_hur_fcst(name,path.format(date=date))
            if r is not None:
                results.append(r)
        pickle.dump(results,open(f'{outpath}/results.pkl','wb'))
    results = pickle.load(open(f'{outpath}/results.pkl','rb'))
    plt.clf() ; plt.close()
    plt.figure(figsize=(10,5))
    for i,(error,predpath,dates) in enumerate(results):
        #dates = dates[:len(error)]
        plt.plot(dates[0],error[0],'o',color=f'C{i}')
        plt.plot(dates,error,color=f'C{i}')
    #plt.axvline(datetime(2022, 9, 28, 19, 10), color='r', linestyle='--')
    plt.text(datetime(2022, 9, 28, 19, 10), plt.gca().get_ylim()[1], 'Florida Landfall', color='r', va='top')
    plt.xlabel('Date')
    plt.ylabel('Distance (km)')
    plt.title(f'Track Error for {hurname}')
    plt.grid()
    plt.savefig(f'{outpath}/error.png')

    #map
    
    realpath = get_hur_track(hurname)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(6,4))
    ax.set_extent([bbox[1], bbox[3], bbox[2], bbox[0]], crs=ccrs.PlateCarree())  # Setting map extent
    ax.coastlines()
    
    ax.plot([x[1] for x in realpath],[x[0] for x in realpath],label='Acutal',color='black')
    for i,(error,predpath,dates) in enumerate(results):
        ax.plot([x[1] for x in predpath],[x[0] for x in predpath],label=f'{dates[0].strftime("%Y%m%d %HZ")}',color=f'C{i}',alpha=0.75)
        ax.plot([x[1] for x in predpath][0],[x[0] for x in predpath][0],'o',color=f'C{i}')
    

    ax.plot()
    ax.legend()
    ax.set_title(f'Track Error for {hurname}')
    plt.savefig(f'{outpath}/error_map.png',dpi=200)


        

#path = '/fast/evaluation/neocasioquad_673M/outputs/{date}.E3rN.npy'
#path = '/fast/evaluation/hegelquad_333M/outputs/{date}.E3rN.npy'
path = '/fast/evaluation/yunghegelquad_333M/outputs/{date}.E3rN.npy'

#many_fcst('delta',path,[f'2020100{d}00' for d in [3,4,5,6,7,8,9,10]],bbox)

if 0:
    gif = Neovis(f'/fast/to_srv/hur/ian2/')
    gif.make()
    exit()

#many_fcst('ian_hegel',path,[f'202209{d}00' for d in [24,25,26,27,28,29]],bbox)
#many_fcst('ian_hegel',path,[f'202209{d}00' for d in [26,27]],bbox)
#if __name__ == '__main__':
#    for sname,times in get_big_storms().items():
#        if sname not in ['ian']: continue
#        many_fcst(sname,path,times)


best_track_dates = {
    # 'beryl': (datetime(2024, 6, 26, 0, 0),
    # datetime(2024, 7, 11, 0, 0),
    # timedelta(days=15)),
    # 'debby': (datetime(2024, 7, 31, 0, 0),
    # datetime(2024, 8, 10, 0, 0),
    # timedelta(days=9, seconds=43200)),
    # 'ernesto': (datetime(2024, 8, 9, 0, 0),
    # datetime(2024, 8, 20, 12, 0),
    # timedelta(days=11, seconds=43200)),
    # 'francine': (datetime(2024, 9, 6, 0, 0), ##
    # datetime(2024, 9, 14, 0, 0),
    # timedelta(days=7, seconds=43200)),
    # 'helene': (datetime(2024, 9, 21, 0, 0), ##
    # datetime(2024, 9, 28, 12, 0),
    # timedelta(days=6, seconds=64800)),
    'kirk': (datetime(2024, 9, 27, 0, 0), ##
    datetime(2024, 10, 7, 12, 0),
    timedelta(days=9, seconds=64800)),
    # 'milton': (datetime(2024, 10, 4, 0, 0), ##
    # datetime(2024, 10, 10, 18, 0),
    # timedelta(days=6, seconds=64800)),
    # 'oscar': (datetime(2024, 10, 9, 0, 0),
    # datetime(2024, 10, 22, 12, 0),
    # timedelta(days=13))
}


start = '/huge/deep/evaluation/ultralatentbachelor_168M/outputs/2022092700.qRCd.npy'
output_path = "/huge/deep/realtime/outputs/WeatherMesh"#-backtest"

for storm in best_track_dates:
    start_datestr = best_track_dates[storm][0].strftime("%Y%m%d") + "00"
    curr = best_track_dates[storm][0]
    curr = datetime(2024, 9, 29, 0, 0)
    while curr <= best_track_dates[storm][1]:
        plot_hur_fcst('weathermesh2', storm, output_path, curr.strftime("%Y%m%d%H"), rerun=False)
        curr += timedelta(hours=24)
    #plot_hur_fcst('weathermesh2', storm, output_path, start_datestr, rerun=False)