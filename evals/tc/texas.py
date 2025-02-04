from evals.tc.tclib import *
from evals.tc.tc import *


AUSTIN = 30.258830, -97.746153
bbox = [37.2, -108.5, 25.5, -92.6]

def get_temp(f):
    xx, mesh = load_instance(f,bbox=bbox)
    x = xx[:, :, mesh.full_varlist.index('167_2t')] - 273.15
    if 0:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(7,4))
        ax.set_extent([bbox[1], bbox[3], bbox[2], bbox[0]], crs=ccrs.PlateCarree())  # Setting map extent
        ax.add_feature(cfeature.STATES)
        ax.coastlines()
        forecast_hour = get_forecast_hour(f)
        im = ax.imshow(x, extent=[bbox[1], bbox[3], bbox[2], bbox[0]], origin='upper', transform=ccrs.PlateCarree())
        plt.colorbar(im,ax=ax,shrink=0.5,aspect=10)
        plt.savefig(f'{outpath}/forecast_hour={forecast_hour:03},_neovis.png')
        plt.clf()
        plt.close()
    t = get_point_nearest(x,mesh,*AUSTIN)
    valid_at = get_valid_at(f)
    return t, valid_at

def get_era5_temp(date):
    x,mesh = load_era5_instance(date,bbox=bbox)
    xa = x[:, :, mesh.full_varlist.index('167_2t')] - 273.15
    t = get_point_nearest(xa,mesh,*AUSTIN).item()
    print(date,t)
    return t

outpath = f'/fast/to_srv/texas/'


cdates = [datetime(2021,2,11) + timedelta(hours=i) for i in range(0,24*8,6)]
ctemps = [get_era5_temp(d) for d in cdates]

print(ctemps)
print(cdates)

os.makedirs(outpath,exist_ok=True)
plt.figure(figsize=(12,5))

for i,date in enumerate(['20210210','20210211','20210212','20210213','20210214','20210215']):
    print(date)
    path = '/fast/evaluation/neocasioquad_673M/outputs/{date}.E3rN.npy'.format(date=date)
    run=get_forecast_run(path)
    rets = [get_temp(f) for f in run]
    os.makedirs(f'{outpath}/austin/',exist_ok=True)
    forecast_at = get_forecast_at(run[0])
    temps = [x[0] for x in rets]
    times = [x[1] for x in rets]
    plt.plot(times,temps,c=f'C{i}')
    plt.plot(times[0],temps[0],'o',c=f'C{i}')
plt.plot(cdates,ctemps,'--',c='black',label='ERA-5')

plt.grid()
plt.ylabel('Temperature (C)')
plt.title(f'Austin Temperature from Feb 2021 Cold Snap | WARNING: DATA EXAMPLE FROM TRAINING DATASET')
plt.legend()
plt.savefig(f'{outpath}/austin/all.png',dpi=150)

#gif = Neovis(outpath)
#gif.make()
