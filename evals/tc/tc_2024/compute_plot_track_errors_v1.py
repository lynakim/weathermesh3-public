import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
from datetime import datetime,timedelta
import os
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cblind as cb
import matplotlib
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.patches as patches
import scipy
import xarray as xr
import sys
import pandas as pd
import tropycal.tracks as tracks
import datetime as dt
from datetime import datetime as dt, timedelta
import constants
import scipy.interpolate as interp
import scipy.ndimage as ndimage
from geopy.distance import geodesic
############################################################
###### FUNCTIONS FOR COMPUTING AND PLOTTING ################
############################################################
############################################
def get_cone_info(storm,forecast):
  #forecast = datetime(2024,9,23,00)
  nhc_forecasts = storm.forecast_dict['OFCL']
  carq_forecasts = storm.forecast_dict['CARQ']
  nhc_forecast_init = [k for k in nhc_forecasts.keys()]
  carq_forecast_init = [k for k in carq_forecasts.keys()]
  nhc_forecast_init_dt = [dt.strptime(k, '%Y%m%d%H') for k in nhc_forecast_init]
  time_diff = np.array([(i - forecast).days + (i - forecast).seconds / 86400 for i in nhc_forecast_init_dt])
  closest_idx = np.abs(time_diff).argmin()
  forecast_dict = nhc_forecasts[nhc_forecast_init[closest_idx]]
  advisory_num = closest_idx + 1
  # Get observed track as per NHC analyses
  track_dict = {
            'lat': [],
            'lon': [],
            'vmax': [],
            'type': [],
            'mslp': [],
            'time': [],
            'extra_obs': [],
            'special': [],
            'ace': 0.0,
        }
  use_carq = True
  year = 2024
  for k in nhc_forecast_init:
    hrs = nhc_forecasts[k]['fhr']
    hrs_carq = carq_forecasts[k]['fhr'] if k in carq_forecast_init else [
	]
	# Account for old years when hour 0 wasn't included directly
	# if 0 not in hrs and k in carq_forecast_init and 0 in hrs_carq:
    if storm.year < 2000 and k in carq_forecast_init and 0 in hrs_carq:
      use_carq = True
      hr_idx = hrs_carq.index(0)
      track_dict['lat'].append(carq_forecasts[k]['lat'][hr_idx])
      track_dict['lon'].append(carq_forecasts[k]['lon'][hr_idx])
      track_dict['vmax'].append(carq_forecasts[k]['vmax'][hr_idx])
      track_dict['mslp'].append(np.nan)
      track_dict['time'].append(carq_forecasts[k]['init'])
      itype = carq_forecasts[k]['type'][hr_idx]
      if itype == "":
        itype = get_storm_type(carq_forecasts[k]['vmax'][0], False)
      track_dict['type'].append(itype)
      hr = carq_forecasts[k]['init'].strftime("%H%M")
      track_dict['extra_obs'].append(0) if hr in [
			'0300', '0900', '1500', '2100'] else track_dict['extra_obs'].append(1)
      track_dict['special'].append("")

    else:
      use_carq = False
      if 3 in hrs:
        hr_idx = hrs.index(3)
        hr_add = 3
      else:
        hr_idx = 0
        hr_add = 0
      track_dict['lat'].append(nhc_forecasts[k]['lat'][hr_idx])
      track_dict['lon'].append(nhc_forecasts[k]['lon'][hr_idx])
      track_dict['vmax'].append(nhc_forecasts[k]['vmax'][hr_idx])
      track_dict['mslp'].append(np.nan)
      track_dict['time'].append(
			nhc_forecasts[k]['init'] + timedelta(hours=hr_add))

      itype = nhc_forecasts[k]['type'][hr_idx]
      if itype == "":
        itype = get_storm_type(nhc_forecasts[k]['vmax'][0], False)
      track_dict['type'].append(itype)

      hr = nhc_forecasts[k]['init'].strftime("%H%M")
      track_dict['extra_obs'].append(0) if hr in [
                    '0300', '0900', '1500', '2100'] else track_dict['extra_obs'].append(1)
      track_dict['special'].append("")

  # Add main elements from storm dict
  for key in ['id', 'operational_id', 'name', 'year']:
    track_dict[key] = storm.dict[key]

  # Add carq to forecast dict as hour 0, if available
  if use_carq and forecast_dict['init'] in track_dict['time']:
    insert_idx = track_dict['time'].index(forecast_dict['init'])
    if 0 in forecast_dict['fhr']:
      forecast_dict['lat'][0] = track_dict['lat'][insert_idx]
      forecast_dict['lon'][0] = track_dict['lon'][insert_idx]
      forecast_dict['vmax'][0] = track_dict['vmax'][insert_idx]
      forecast_dict['mslp'][0] = track_dict['mslp'][insert_idx]
      forecast_dict['type'][0] = track_dict['type'][insert_idx]
    else:
     forecast_dict['fhr'].insert(0, 0)
     forecast_dict['lat'].insert(0, track_dict['lat'][insert_idx])
     forecast_dict['lon'].insert(0, track_dict['lon'][insert_idx])
     forecast_dict['vmax'].insert(0, track_dict['vmax'][insert_idx])
     forecast_dict['mslp'].insert(0, track_dict['mslp'][insert_idx])
     forecast_dict['type'].insert(0, track_dict['type'][insert_idx])
  forecast_dict['advisory_num'] = advisory_num
  forecast_dict['basin'] = storm.basin
  return forecast_dict
######################################
def get_wm_track(data,init_date):
  output_data = {}
  output_data['lats'] = []
  output_data['lons'] = []
  output_data['times'] = []
  output_data['fhr'] = []
  for i,k in enumerate(data):
    if 'valid_at' in k.keys():
      output_data['lats'].append(k['latitude'])
      output_data['lons'].append(k['longitude'])
      output_data['times'].append(datetime.strptime(k['time'].split('+')[0],'%Y-%m-%dT%H:%M:%S').strftime('%Y%m%d%H'))
      output_data['fhr'].append(int((datetime.strptime(k['time'].split('+')[0],'%Y-%m-%dT%H:%M:%S') - init_date).total_seconds()/60./60.))
  return output_data
#######################################
def get_wm_track_id(wm_data,truth_lat,truth_lon,truth_index):
  keep_track = None
  for j,k in enumerate(wm_data):
    track_index = np.array([d for d in range(0, len(wm_data[k])) if 'valid_at' in wm_data[k][d]])
    if track_index.shape[0] == 0:
      continue
    else:
      track_index = track_index[0]
    distance = geodesic((truth_lat[truth_index],truth_lon[truth_index]), (wm_data[k][track_index]['latitude'],wm_data[k][track_index]['longitude'])).km
    if distance < 500:
      keep_track = k
      break
  return keep_track
##################
def create_map(search_limits=[None, None, None, None]):
	################################
	prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9),
                            'dpi': 200, 'plot_gridlines': True}
	fig = plt.figure(figsize=(8,8))
	proj = ccrs.PlateCarree(central_longitude=0)
	ax = plt.subplot(1, 1, 1, projection=proj)
	ax.set_extent([-98, -54,10,40], crs=proj)
	res = '50m'
	ocean_mask = ax.add_feature(cfeature.OCEAN.with_scale(res), facecolor=prop['ocean_color'], edgecolor='face', zorder=0)
	lake_mask = ax.add_feature(cfeature.LAKES.with_scale(res), facecolor=prop['ocean_color'], edgecolor='face', zorder=0)
	continent_mask = ax.add_feature(cfeature.LAND.with_scale(res), facecolor=prop['land_color'], edgecolor='face', zorder=1)
	
	states = ax.add_feature(cfeature.STATES.with_scale(res), linewidths=prop['linewidth'], linestyle='solid', edgecolor=prop['linecolor'],alpha=1.0, zorder=1)
	countries = ax.add_feature(cfeature.BORDERS.with_scale(res), linewidths=prop['linewidth'], linestyle='solid', edgecolor=prop['linecolor'],zorder=2)
	coastlines = ax.add_feature(cfeature.COASTLINE.with_scale(res), linewidths=prop['linewidth'], linestyle='solid', edgecolor=prop['linecolor'],zorder=3)
	return ax
#######################################################################
##### END FUNCTIONS ###################################################
#######################################################################
#######################################################################
## get tropycal data
basin = tracks.TrackDataset(basin='north_atlantic',source='hurdat',include_btk=True)

storm = basin.get_storm(('milton',2024))
storm_dict = storm.to_dict()
truth_lat = storm_dict['lat'][:];truth_lon = storm_dict['lon'][:]
truth_time = storm_dict['time'][:]
try:
	storm.forecast_dict
except:
	storm.get_operational_forecasts()
##############################################################################
######
# WM track
init_times = pd.date_range(start=datetime(2024,10,6,0),end=datetime(2024,10,9,12),freq='6h').to_pydatetime()

wm_track = {}
for i in range(0,init_times.shape[0]):
  truth_index = np.where(init_times[i] == np.array(truth_time))[0][0]
  print('Getting WM tracks => {0}'.format(init_times[i]))
  wm_data = json.load(open('/Users/criedel/mnt/wb-dlnwp/viz/WeatherMesh/tcs/{0}_tcs.Qfiz.json'.format(init_times[i].strftime('%Y%m%d%H'))))
  keep_track = get_wm_track_id(wm_data,truth_lat,truth_lon,truth_index)
  if keep_track == None:
    output_data = {}
    output_data['lats'] = []
    output_data['lons'] = []
    output_data['times'] = []
    output_data['fhr'] = []
    wm_track[int(init_times[i].strftime('%Y%m%d%H'))] = output_data
  else:
    print(int(init_times[i].strftime('%Y%m%d%H')))
    wm_track[int(init_times[i].strftime('%Y%m%d%H'))] = get_wm_track(wm_data[keep_track],init_times[i])


cone_prop = {'plot': True, 'linewidth': 1.5, 'linecolor': 'k', 'alpha': 0.2,
                             'days': 5, 'fillcolor': 'category', 'label_category': True, 'ms': 12}

wm_track_err = {}
nhc_track_err = {}
fhr = np.arange(0,168+6,6)
for f in range(0,fhr.shape[0]):
  wm_track_err[fhr[f]] = []
  nhc_track_err[fhr[f]] = []
truth_times_int = np.array([int(d.strftime('%Y%m%d%H')) for d in truth_time])

plt.rc('font', weight='bold')
for i in range(0,init_times.shape[0]):
  #########
  ## Get NHC forecast for initialization time
  forecast_dict = get_cone_info(storm,init_times[i])
  #################
  ### Compute WM track error
  wm_lat = wm_track[int(init_times[i].strftime('%Y%m%d%H'))]['lats']
  wm_lon = wm_track[int(init_times[i].strftime('%Y%m%d%H'))]['lons']
  wm_fhr = wm_track[int(init_times[i].strftime('%Y%m%d%H'))]['fhr']
  wm_vtime = np.array(wm_track[int(init_times[i].strftime('%Y%m%d%H'))]['times']).astype('int')
  ###
  for v in range(0,len(wm_vtime)):
    indt = np.where(wm_vtime[v] == truth_times_int)[0]
    if indt.shape[0] == 0:
      continue
    distance = geodesic((truth_lat[indt[0]],truth_lon[indt[0]]), (wm_lat[v],wm_lon[v])).km
    wm_track_err[int(wm_fhr[v])].append(distance)
  #####################################
  #### Compute NHC track error
  nhc_fhr = forecast_dict['fhr']
  nhc_vtime = np.array([int((forecast_dict['init'] + timedelta(hours=int(d))).strftime('%Y%m%d%H')) for d in forecast_dict['fhr']])
  #### NHC forecasts have less freq forecast hours, decided to interp to every 6-hours,not required
  nhc_lat = np.interp(np.arange(0,forecast_dict['fhr'][-1]+6,6),nhc_fhr,forecast_dict['lat'],left=np.nan,right=np.nan)
  nhc_lon = np.interp(np.arange(0,forecast_dict['fhr'][-1]+6,6),nhc_fhr,forecast_dict['lon'],left=np.nan,right=np.nan)
  nhc_vtime = np.array([int((forecast_dict['init'] + timedelta(hours=int(d))).strftime('%Y%m%d%H')) for d in np.arange(0,forecast_dict['fhr'][-1]+6,6)])
  nhc_fhr = np.arange(0,forecast_dict['fhr'][-1]+6,6)
  if forecast_dict['init'] != init_times[i]:
    continue
  for v in range(0,nhc_lat.shape[0]):
    if np.isnan(nhc_lat[v]) or np.isnan(nhc_lon[v]):
      continue
    indt = np.where(nhc_vtime[v] == truth_times_int)[0]
    if indt.shape[0] == 0:
      continue
    distance = geodesic((truth_lat[indt[0]],truth_lon[indt[0]]), (nhc_lat[v],nhc_lon[v])).km
    nhc_track_err[int(nhc_fhr[v])].append(distance)
  #######################################

data_holder = {str(k):nhc_track_err[k] for i,k in enumerate(nhc_track_err)}
json.dump(data_holder,open('helene_nhc_trackerr.json','w'))
data_holder = {str(k):wm_track_err[k] for i,k in enumerate(wm_track_err)}
json.dump(data_holder,open('helene_wm_trackerr.json','w'))


wm_mean_track_err = np.zeros((fhr.shape[0]));wm_mean_samples = np.zeros((fhr.shape[0]))
nhc_mean_track_err = np.zeros((fhr.shape[0]));nhc_mean_samples = np.zeros((fhr.shape[0]))
for f in range(0,fhr.shape[0]):
  if len(wm_track_err[fhr[f]]) == 0:
    wm_mean_track_err[f] = np.nan
  else:
    wm_mean_track_err[f] = np.nanmean(wm_track_err[fhr[f]])
    wm_mean_samples[f] = len(wm_track_err[fhr[f]])
  ###############################
  if len(nhc_track_err[fhr[f]]) == 0:
    nhc_mean_track_err[f] = np.nan
  else:
    nhc_mean_track_err[f] = np.nanmean(nhc_track_err[fhr[f]])
    nhc_mean_samples[f] = len(nhc_track_err[fhr[f]])

index_end = max(np.where(wm_mean_samples==0)[0][0],np.where(nhc_mean_samples==0)[0][0])
max_val = max(np.nanmax(nhc_mean_track_err),np.nanmax(wm_mean_track_err))+10

fig,ax = plt.subplots(1,1,figsize=(8,4))
for z in range(1,index_end):
  ax.text(fhr[z],-25,'{0}'.format(int(wm_mean_samples[z])),fontsize=5,ha='center',color='red')
  ax.text(fhr[z],-32,'{0}'.format(int(nhc_mean_samples[z])),fontsize=5,ha='center',color='blue')

ax.plot(fhr,nhc_mean_track_err,'-ob',markersize=3,label='National Hurricane Center',zorder=3)
ax.plot(fhr,wm_mean_track_err,'-or',markersize=3,label='Windborne WeatherMesh',zorder=3)
ax.grid(True,linestyle='dashed',linewidth=0.5,axis='both')
ax.set_ylabel('Track Error (km)',weight='bold',fontsize=9)
ax.legend()
ax.set_ylim(0,max_val)
ax.set_title('Average Ground Track Errors for Milton',weight='bold')
ax.set_xlabel('Forecast Lead Time (hours)\nForecast Samples',weight='bold',labelpad=10,fontsize=9)
ax.set_xlim(6,fhr[index_end-1]+3);ax.set_xticks(np.arange(6,fhr[index_end-1]+6,6))
ax.tick_params(axis='both', labelsize=8)

plt.savefig('track_err.jpg',dpi=550,bbox_inches='tight')
os.system('open track_err.jpg')


