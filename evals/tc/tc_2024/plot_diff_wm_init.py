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
def plug_array(small, large, small_coords, large_coords):
    r"""
    Plug small array into large array with matching coords.

    Parameters
    ----------
    small : numpy.ndarray
        Small array to be plugged into the larger array.
    large : numpy.ndarray
        Large array for the small array to be plugged into.
    small_coords : dict
        Dictionary containing 'lat' and 'lon' keys, whose values are numpy.ndarrays of lat & lon for the small array.
    large_coords : dict
        Dictionary containing 'lat' and 'lon' keys, whose values are numpy.ndarrays of lat & lon for the large array.

    Returns
    -------
    numpy.ndarray
        An array of the same dimensions as "large", with the small array plugged inside the large array.
    """

    small_lat = np.round(small_coords['lat'], 2)
    small_lon = np.round(small_coords['lon'], 2)
    large_lat = np.round(large_coords['lat'], 2)
    large_lon = np.round(large_coords['lon'], 2)

    small_minlat = np.nanmin(small_lat)
    small_maxlat = np.nanmax(small_lat)
    small_minlon = np.nanmin(small_lon)
    small_maxlon = np.nanmax(small_lon)

    if small_minlat in large_lat:
        minlat = np.where(large_lat == small_minlat)[0][0]
    else:
        minlat = min(large_lat)
    if small_maxlat in large_lat:
        maxlat = np.where(large_lat == small_maxlat)[0][0]
    else:
        maxlat = max(large_lat)
    if small_minlon in large_lon:
        minlon = np.where(large_lon == small_minlon)[0][0]
    else:
        minlon = min(large_lon)
    if small_maxlon in large_lon:
        maxlon = np.where(large_lon == small_maxlon)[0][0]
    else:
        maxlon = max(large_lon)

    large[minlat:maxlat+1, minlon:maxlon+1] = small

    return large
#######################################
def nhc_cone_radii(year, basin, forecast_hour=None):
    r"""
    Retrieve the official NHC Cone of Uncertainty radii by basin, year and forecast hour(s). Units are in nautical miles.

    Parameters
    ----------
    year : int
        Valid year for cone of uncertainty radii.
    basin : str
        Basin for cone of uncertainty radii. If basin is invalid, return value will be an empty dict. Please refer to :ref:`options-domain` for available basin options.
    forecast_hour : int or list, optional
        Forecast hour(s) to retrieve the cone of uncertainty for. If empty, all available forecast hours will be retrieved.

    Returns
    -------
    dict
        Dictionary with forecast hour(s) as the keys, and the cone radius in nautical miles for each respective forecast hour as the values.

    Notes
    -----
    1. NHC cone radii are available beginning 2008 onward. Radii for years before 2008 will be defaulted to 2008, and if the current year's radii are not available yet, the radii for the most recent year will be returned.

    2. NHC began producing cone radii for forecast hour 60 in 2020. Years before 2020 do not have a forecast hour 60.
    """

    # Source: https://www.nhc.noaa.gov/verification/verify3.shtml
    # Source 2: https://www.nhc.noaa.gov/aboutcone.shtml
    # Radii are in nautical miles
    cone_climo_hr = [3, 12, 24, 36, 48, 72, 96, 120]

    # Basin check
    if basin not in ['north_atlantic', 'east_pacific']:
        return {}

    # Fix for 2020 and later that incorporates 60 hour forecasts
    if year >= 2020:
        cone_climo_hr = [3, 12, 24, 36, 48, 60, 72, 96, 120]

    # Forecast hour check
    if forecast_hour is None:
        forecast_hour = cone_climo_hr
    elif isinstance(forecast_hour, int):
        if forecast_hour not in cone_climo_hr:
            raise ValueError(
                f"Forecast hour {forecast_hour} is invalid. Available forecast hours for {year} are: {cone_climo_hr}")
        else:
            forecast_hour = [forecast_hour]
    elif isinstance(forecast_hour, list):
        forecast_hour = [i for i in forecast_hour if i in cone_climo_hr]
        if len(forecast_hour) == 0:
            raise ValueError(
                f"Requested forecast hours are invalid. Available forecast hours for {year} are: {cone_climo_hr}")
    else:
        raise TypeError("forecast_hour must be of type int or list")

    # Year check
    if year > np.max([k for k in constants.CONE_SIZE_ATL.keys()]):
        year = [k for k in constants.CONE_SIZE_ATL.keys()][0]
        warnings.warn(
            f"No cone information is available for the requested year. Defaulting to {year} cone.")
    elif year not in constants.CONE_SIZE_ATL.keys():
        year = 2008
        warnings.warn(
            "No cone information is available for the requested year. Defaulting to 2008 cone.")

    # Retrieve data
    cone_radii = {}
    for hour in list(np.sort(forecast_hour)):
        hour_index = cone_climo_hr.index(hour)
        if basin == 'north_atlantic':
            cone_radii[hour] = constants.CONE_SIZE_ATL[year][hour_index]
        elif basin == 'east_pacific':
            cone_radii[hour] = constants.CONE_SIZE_PAC[year][hour_index]

    return cone_radii
#######################################
def generate_nhc_cone(forecast, basin, shift_lons=False, cone_days=5, cone_year=None, grid_res=0.05, return_xarray=False):
    r"""
    Generates a gridded cone of uncertainty using forecast data from NHC.

    Parameters
    ----------
    forecast : dict
        Dictionary containing forecast data
    basin : str
        Basin for cone of uncertainty radii. Please refer to :ref:`options-domain` for available basin options.
    
    Other Parameters
    ----------------
    shift_lons : bool, optional
        If true, grid will be shifted to +0 to +360 degrees longitude. Default is False (-180 to +180 degrees).
    cone_days : int, optional
        Number of forecast days to generate the cone through. Default is 5 days.
    cone_year : int, optional
        Year valid for cone radii. If None, this fuction will attempt to retrieve the year from the forecast dict.
    grid_res : int or float, optional
        Horizontal resolution of the cone of uncertainty grid in degrees. Default is 0.05 degrees.
    return_xarray : bool, optional
        If True, returns output as an xarray Dataset. Default is False, returning output as a dictionary.

    Returns
    -------
    dict or xarray.Dataset
        Depending on `return_xarray`, returns either a dictionary or an xarray Dataset containing the gridded cone of uncertainty and its accompanying attributes.

    Notes
    -----
    Forecast dicts can be retrieved for realtime storm objects using ``RealtimeStorm.get_forecast_realtime()``, and for archived storms using ``Storm.get_nhc_forecast_dict()``.
    """

    # Check forecast dict has all required keys
    check_dict = [True if i in forecast.keys() else False for i in [
        'fhr', 'lat', 'lon', 'init']]
    if False in check_dict:
        raise ValueError(
            "forecast dict must contain keys 'fhr', 'lat', 'lon' and 'init'. You may retrieve a forecast dict for a Storm object through 'storm.get_operational_forecasts()'.")

    # Check forecast basin
    if basin not in constants.ALL_BASINS:
        raise ValueError("basin cannot be identified.")

    # Retrieve cone of uncertainty year
    if cone_year is None:
        cone_year = forecast['init'].year
    if cone_year > np.max([k for k in constants.CONE_SIZE_ATL.keys()]):
        cone_year = [k for k in constants.CONE_SIZE_ATL.keys()][0]
        warnings.warn(
            f"No cone information is available for the requested year. Defaulting to {cone_year} cone.")
    elif cone_year not in constants.CONE_SIZE_ATL.keys():
        cone_year = 2008
        warnings.warn(
            "No cone information is available for the requested year. Defaulting to 2008 cone.")

    # Retrieve cone size and hours for given year
    if basin in ['north_atlantic', 'east_pacific']:
        output = nhc_cone_radii(cone_year, basin)
        cone_climo_hr = [k for k in output.keys()]
        cone_size = [output[k] for k in output.keys()]
    else:
        cone_climo_hr = [3, 12, 24, 36, 48, 72, 96, 120]
        cone_size = 0

    # Function for interpolating between 2 times
    def temporal_interpolation(value, orig_times, target_times):
        f = interp.interp1d(orig_times, value)
        ynew = f(target_times)
        return ynew

    # Function for finding nearest value in an array
    def find_nearest(array, val):
        return array[np.abs(array - val).argmin()]

    # Function for adding a radius surrounding a point
    def add_radius(lats, lons, vlat, vlon, rad, grid_res=0.05):

        # construct new array expanding slightly over rad from lat/lon center
        grid_fac = (rad*4)/111.0

        # Make grid surrounding position coordinate & radius of circle
        nlon = np.arange(find_nearest(lons, vlon-grid_fac),
                         find_nearest(lons, vlon+grid_fac+grid_res), grid_res)
        nlat = np.arange(find_nearest(lats, vlat-grid_fac),
                         find_nearest(lats, vlat+grid_fac+grid_res), grid_res)
        lons, lats = np.meshgrid(nlon, nlat)
        return_arr = np.zeros((lons.shape))

        # Calculate distance from vlat/vlon at each gridpoint
        r_earth = 6.371 * 10**6
        dlat = np.subtract(np.radians(lats), np.radians(vlat))
        dlon = np.subtract(np.radians(lons), np.radians(vlon))

        a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lats)) * \
            np.cos(np.radians(vlat)) * np.sin(dlon/2) * np.sin(dlon/2)
        c = 2 * np.arctan(np.sqrt(a), np.sqrt(1-a))
        dist = (r_earth * c)/1000.0
        dist = dist * 0.621371  # to miles
        dist = dist * 0.868976  # to nautical miles

        # Mask out values less than radius
        return_arr[dist <= rad] = 1

        # Attach small array into larger subset array
        small_coords = {'lat': nlat, 'lon': nlon}

        return return_arr, small_coords

    # --------------------------------------------------------------------

    # Check if fhr3 is available, then get forecast data
    flag_12 = 0
    if forecast['fhr'][0] == 12:
        flag_12 = 1
        cone_climo_hr = cone_climo_hr[1:]
        fcst_lon = forecast['lon']
        fcst_lat = forecast['lat']
        fhr = forecast['fhr']
        t = np.array(forecast['fhr'])/6.0
        subtract_by = t[0]
        t = t - t[0]
        interp_fhr_idx = np.arange(t[0], t[-1]+0.1, 0.1) - t[0]
    elif 3 in forecast['fhr'] and 1 in forecast['fhr'] and 0 in forecast['fhr']:
        fcst_lon = forecast['lon'][2:]
        fcst_lat = forecast['lat'][2:]
        fhr = forecast['fhr'][2:]
        t = np.array(fhr)/6.0
        interp_fhr_idx = np.arange(t[0], t[-1]+0.01, 0.1)
    elif 3 in forecast['fhr'] and 0 in forecast['fhr']:
        idx = np.array([i for i, j in enumerate(
            forecast['fhr']) if j in cone_climo_hr])
        fcst_lon = np.array(forecast['lon'])[idx]
        fcst_lat = np.array(forecast['lat'])[idx]
        fhr = np.array(forecast['fhr'])[idx]
        t = np.array(fhr)/6.0
        interp_fhr_idx = np.arange(t[0], t[-1]+0.01, 0.1)
    elif forecast['fhr'][1] < 12:
        cone_climo_hr[0] = 0
        fcst_lon = [forecast['lon'][0]]+forecast['lon'][2:]
        fcst_lat = [forecast['lat'][0]]+forecast['lat'][2:]
        fhr = [forecast['fhr'][0]]+forecast['fhr'][2:]
        t = np.array(fhr)/6.0
        interp_fhr_idx = np.arange(t[0]/6.0, t[-1]+0.1, 0.1)
    else:
        cone_climo_hr[0] = 0
        fcst_lon = forecast['lon']
        fcst_lat = forecast['lat']
        fhr = forecast['fhr']
        t = np.array(fhr)/6.0
        interp_fhr_idx = np.arange(t[0], t[-1]+0.1, 0.1)

    # Determine index of forecast day cap
    if (cone_days*24) in fhr:
        cone_day_cap = list(fhr).index(cone_days*24)+1
        fcst_lon = fcst_lon[:cone_day_cap]
        fcst_lat = fcst_lat[:cone_day_cap]
        fhr = fhr[:cone_day_cap]
        t = np.array(fhr)/6.0
        interp_fhr_idx = np.arange(interp_fhr_idx[0], t[-1]+0.1, 0.1)
    else:
        cone_day_cap = len(fhr)

    # Account for dateline
    if shift_lons:
        temp_lon = np.array(fcst_lon)
        temp_lon[temp_lon < 0] = temp_lon[temp_lon < 0]+360.0
        fcst_lon = temp_lon.tolist()

    # Interpolate forecast data temporally and spatially
    interp_kind = 'quadratic'
    if len(t) == 2:
        interp_kind = 'linear'  # Interpolate linearly if only 2 forecast points
    x1 = interp.interp1d(t, fcst_lon, kind=interp_kind)
    y1 = interp.interp1d(t, fcst_lat, kind=interp_kind)
    interp_fhr = interp_fhr_idx * 6
    interp_lon = x1(interp_fhr_idx)
    interp_lat = y1(interp_fhr_idx)

    # Return if no cone specified
    if cone_size == 0:
        return_dict = {'center_lon': interp_lon, 'center_lat': interp_lat}
        if return_xarray:
            import xarray as xr
            return xr.Dataset(return_dict)
        else:
            return return_dict

    # Interpolate cone radius temporally
    cone_climo_hr = cone_climo_hr[:cone_day_cap]
    cone_size = cone_size[:cone_day_cap]

    cone_climo_fhrs = np.array(cone_climo_hr)
    if flag_12 == 1:
        interp_fhr += (subtract_by*6.0)
        cone_climo_fhrs = cone_climo_fhrs[1:]
    idxs = np.nonzero(np.in1d(np.array(fhr), np.array(cone_climo_hr)))
    temp_arr = np.array(cone_size)[idxs]
    interp_rad = np.apply_along_axis(lambda n: temporal_interpolation(
        n, fhr, interp_fhr), axis=0, arr=temp_arr)

    # Initialize 0.05 degree grid
    gridlats = np.arange(min(interp_lat)-7, max(interp_lat)+7, grid_res)
    gridlons = np.arange(min(interp_lon)-7, max(interp_lon)+7, grid_res)
    gridlons2d, gridlats2d = np.meshgrid(gridlons, gridlats)

    # Iterate through fhr, calculate cone & add into grid
    large_coords = {'lat': gridlats, 'lon': gridlons}
    griddata = np.zeros((gridlats2d.shape))
    for i, (ilat, ilon, irad) in enumerate(zip(interp_lat, interp_lon, interp_rad)):
        temp_grid, small_coords = add_radius(
            gridlats, gridlons, ilat, ilon, irad, grid_res=grid_res)
        plug_grid = np.zeros((griddata.shape))
        plug_grid = plug_array(temp_grid, plug_grid,
                               small_coords, large_coords)
        griddata = np.maximum(griddata, plug_grid)

    if return_xarray:
        import xarray as xr
        cone = xr.DataArray(griddata, coords=[gridlats, gridlons], dims=[
                            'grid_lat', 'grid_lon'])
        return_ds = {
            'cone': cone,
            'center_lon': interp_lon,
            'center_lat': interp_lat
        }
        return_ds = xr.Dataset(return_ds)
        return_ds.attrs['year'] = cone_year
        return return_ds

    else:
        return_dict = {'lat': gridlats, 'lon': gridlons, 'lat2d': gridlats2d, 'lon2d': gridlons2d, 'cone': griddata,
                       'center_lon': interp_lon, 'center_lat': interp_lat, 'year': cone_year}
        return return_dict
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
def get_wm_track_id(wm_data):
  keep_track = None
  for j,k in enumerate(wm_data):
    track_index = np.array([d for d in range(0, len(wm_data[k])) if 'valid_at' in wm_data[k][d]])
    if track_index.shape[0] == 0:
      continue
    else:
      track_index = track_index[0]
    distance = distance = geodesic((truth_lat[truth_index],truth_lon[truth_index]), (wm_data[k][track_index]['latitude'],wm_data[k][track_index]['longitude'])).km
    print(distance)
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
truth_lat = storm_dict['lat'][10:];truth_lon = storm_dict['lon'][10:]
truth_time = storm_dict['time'][10:]
try:
	storm.forecast_dict
except:
	storm.get_operational_forecasts()
##############################################################################
######
# WM track
init_times = pd.date_range(start=datetime(2024,10,6,12),end=datetime(2024,10,9,12),freq='6h').to_pydatetime()
wm_track = {}
for i in range(0,init_times.shape[0]):
  print(i)
  truth_index = np.where(init_times[i] == np.array(truth_time))[0][0]
  print('Getting WM tracks => {0}'.format(init_times[i]))
  wm_data = json.load(open('/Users/criedel/mnt/wb-dlnwp/viz/WeatherMesh/tcs/{0}_tcs.Qfiz.json'.format(init_times[i].strftime('%Y%m%d%H'))))
  keep_track = get_wm_track_id(wm_data)
  if keep_track == None:
    output_data = {}
    output_data['lats'] = []
    output_data['lons'] = []
    output_data['times'] = []
    output_data['fhr'] = []
    wm_track[int(init_times[i].strftime('%Y%m%d%H'))] = output
  else:
    print(int(init_times[i].strftime('%Y%m%d%H')))
    wm_track[int(init_times[i].strftime('%Y%m%d%H'))] = get_wm_track(wm_data[keep_track],init_times[i])


cone_prop = {'plot': True, 'linewidth': 1.5, 'linecolor': 'k', 'alpha': 0.2,
                             'days': 5, 'fillcolor': 'category', 'label_category': True, 'ms': 12}

plt.rc('font', weight='bold')
for i in range(0,init_times.shape[0]):
  #########
  forecast_dict = get_cone_info(storm,init_times[i])
  cone = generate_nhc_cone(forecast_dict,forecast_dict['basin'],cone_days=7)
  cone_2d = cone['cone']
  cone_2d = ndimage.gaussian_filter(cone_2d, sigma=0.5, order=0)
  #########

  date = int(init_times[i].strftime('%Y%m%d%H'))
  fig = plt.figure(figsize=(8,8))
  ax1 = create_map()
  l1 = ax1.plot(storm_dict['lon'][10:],storm_dict['lat'][10:],'-ko',linewidth=2.0,label='Actual Ground Track',transform=ccrs.PlateCarree(central_longitude=0),markersize=3)
  l2 = ax1.plot(wm_track[date]['lons'],wm_track[date]['lats'],'-ro',linewidth=2.0,label='Windborne WeatherMesh(WM)',transform=ccrs.PlateCarree(central_longitude=0),markersize=3)
  l3 = ax1.plot(forecast_dict['lon'],forecast_dict['lat'],'-bo',linewidth=1.5,label='National Hurricane Center(NHC)',transform=ccrs.PlateCarree(central_longitude=0),markersize=2.75,alpha=0.9)
  caf = ax1.contourf(cone['lon2d'], cone['lat2d'], cone_2d, [0.9, 1.1], colors=['b', 'b'], alpha=cone_prop['alpha'], zorder=4, transform=ccrs.PlateCarree(),label='NHC Uncertainty Cone')
  handles_filled,labels_filled = caf.legend_elements()
  ax1.contour(cone['lon2d'], cone['lat2d'], cone_2d, [0.9], linewidths=0.5, colors=['b'],alpha=0.1, zorder=4, transform=ccrs.PlateCarree())
  #ax1.plot(storm.forecast_dict['AVNO']['2024092312']['lon'],storm.forecast_dict['AVNO']['2024092312']['lat'],'-co',linewidth=2.0,label='NOAA GFS',transform=ccrs.PlateCarree(central_longitude=0),markersize=3)

  #ax1.legend(l1+l2+l3+handles_filled,['Actual Ground Track','Windborne WeatherMesh(WM)','National Hurricane Center(NHC)','NHC Uncertainty Cone'],fontsize=8)
  ax1.set_title("Ground Track for Hurricane Milton - Forecast Initialization: {0}".format(init_times[i].strftime("%Y-%m-%d %H Z")),weight='bold',fontsize=10)
  save_name = 'tracks_{0}.jpg'.format(date)
  plt.savefig(save_name,dpi=550,bbox_inches='tight')
  os.system('open {0}'.format(save_name))
  


