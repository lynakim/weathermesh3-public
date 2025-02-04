# Code to access TomorrowNow Global Access Platform (GAP) API
# https://www.notion.so/windborne/TomorrowNow-Global-Access-Platform-GAP-API-142264d62cec809eb23ad4a303f06abd
import requests
import os
from datetime import datetime, timedelta
import xarray as xr
import warnings
import argparse

ATTRIBUTES_DICT = {
    'salient_seasonal_forecast': [
        'max_temperature','max_temperature_anom','max_temperature_clim','min_temperature','min_temperature_anom','min_temperature_clim',
        'precipitation','precipitation_anom','precipitation_clim', 'relative_humidity','relative_humidity_anom','relative_humidity_clim',
        'solar_radiation','solar_radiation_anom','solar_radiation_clim','temperature','temperature_anom','temperature_clim',
        'wind_speed','wind_speed_anom','wind_speed_clim'
    ],
    'disdrometer_ground_observation': [
        'atmospheric_pressure', 'depth_of_water', 'electrical_conductivity_of_precipitation', 'electrical_conductivity_of_water', 
        'lightning_distance', 'precipitation', 'precipitation_total', 'relative_humidity', 'shortwave_radiation',
        'soil_moisture_content', 'soil_temperature', 'surface_air_temperature', 'wind_gusts', 'wind_heading', 'wind_speed'
    ],
    # 90d api limit. do csv
    'arable_ground_observation': ['max_day_temperature','max_relative_humidity','mean_day_temperature','mean_relative_humidity','min_day_temperature','min_relative_humidity','precipitation','precipitation_total','sea_level_pressure','total_evapotranspiration_flux','wind_heading','wind_speed','wind_speed_max','wind_speed_min'],
    'tahmo_ground_observation': ['average_air_temperature', 'max_air_temperature', 'max_relative_humidity', 'min_air_temperature', 'min_relative_humidity', 'precipitation', 'solar_radiation']
    
}

def get_previous_monday():
  return (datetime.now().date() - timedelta(days=datetime.now().weekday()))

# currently only salient supported, 
def get_tomorrow_gap_data(product='salient_seasonal_forecast', attributes=None, start_date=datetime.today().strftime('%Y-%m-%d'), start_time='00:00:00', end_date=None):
  if attributes is None:
    attributes = ATTRIBUTES_DICT[product]
  attributes = ','.join(attributes)
  if end_date is None:
    end_date = start_date
  end_time = start_time#'12:00:00'
  url = f'https://tngap.sta.do.kartoza.com/api/v1/measurement/?product={product}&attributes={attributes}&start_date={start_date}&start_time={start_time}&end_date={end_date}&end_time={end_time}&output_type=netcdf&altitudes=0%2C10000&bbox=28.25%2C-12%2C42.25%2C5.25'
  headers = {
      'accept': 'application/json',
      'authorization': 'Basic a2FyZW5Ad2luZGJvcm5lc3lzdGVtcy5jb206Z0dnWlUjZjN4aUwySlBQ',
      'X-CSRFToken': 'FifXEOZyDGa6Sok5LkZH4T9ygZoYnzPm3GKbaQPUnPuXGG6vUfUY6qEyjALYQTf7'
  }

  response = requests.get(url, headers=headers)
  return response

def download_salient_90d_forecasts(start_date):
    curr_start_date = start_date
    # loop over 3 30 day periods (including day 0) + 4th file of the 90th day
    for i in range(0, 4):
        fn = f"/tmp/gap_tomorrow/{start_date.strftime('%Y%m%d')}_month_{i}.nc"
        if os.path.exists(fn):
            print(f"Already have {fn}", flush=True)
            continue
        
        print(f"Downloading {curr_start_date.strftime('%Y-%m-%d')} to {(curr_start_date+timedelta(days=29)).strftime('%Y-%m-%d')}", flush=True)
        response = get_tomorrow_gap_data(start_date=curr_start_date.strftime('%Y-%m-%d'), end_date=(curr_start_date+timedelta(days=29)).strftime('%Y-%m-%d'))
        assert response.status_code == 200, f"Error: {response.json()}"

        
        fn_tmp = fn + '.tmp'
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn_tmp, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        os.rename(fn_tmp, fn)
        print(f"Downloaded {fn}", flush=True)
        curr_start_date = (curr_start_date + timedelta(days=30))


def combine_and_save_forecast_stats(start_date, base_path="/huge/proc/gap_tomorrow/salient_seasonal_forecast"):
    """Combine 4 monthly files and save statistics for each variable"""
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Create paths for the 4 monthly files
    monthly_files = [
        f"/tmp/gap_tomorrow/{start_date.strftime('%Y%m%d')}_month_{i}.nc"
        for i in range(4)
    ]
    
    # Read and combine the monthly files
    datasets = []
    for file in monthly_files:
        if os.path.exists(file):
            ds = xr.open_dataset(file)
            datasets.append(ds)
        else:
            print(f"Warning: File not found: {file}", flush=True)
            continue
    
    if not datasets:
        raise FileNotFoundError("No monthly files found")
    
    # Combine along the forecast_date dimension
    combined_ds = xr.concat(datasets, dim='forecast_day')
    
    # Create output folder for this initialization date
    init_folder = os.path.join(base_path, start_date.strftime('%Y%m%d'))
    os.makedirs(init_folder, exist_ok=True)
    
    # Process each variable
    for var in combined_ds.data_vars:
        print(f"Processing {var}...", flush=True)
        
        # Calculate statistics across ensemble dimension
        if var[-5:] == '_clim':
            # check folder above init_folder for climatology
            clim_folder = os.path.join(base_path, "climatology")
            output_path = os.path.join(clim_folder, f"{var}.nc")
            if os.path.exists(output_path):
                print(f"Already have climatology at {output_path}", flush=True)
                continue
            else:
                os.makedirs(clim_folder, exist_ok=True)
                stats_ds = combined_ds[var]
        else:
            var_mean = combined_ds[var].mean(dim='ensemble')
            var_std = combined_ds[var].std(dim='ensemble')
            
            # Create a new dataset with both statistics
            stats_ds = xr.Dataset({
                f"{var}_mean": var_mean,
                f"{var}_std": var_std
            })
            output_path = os.path.join(init_folder, f"{var}.nc")
        stats_ds.attrs["units"] = combined_ds[var].units
        stats_ds.to_netcdf(output_path)
        print(f"Saved {var} to {output_path}", flush=True)
        stats_ds.close()


def read_salient_90d_forecasts(var, start_date=None):
    base_path = "/huge/proc/gap_tomorrow/salient_seasonal_forecast/"
    if var[-5:] == '_clim':
        assert start_date is None, "start_date is not required for climatology variables"
        clim_folder = os.path.join(base_path, "climatology")
        ds = xr.open_dataset(f"{clim_folder}/{var}.nc")
    else:
        assert start_date is not None, "start_date is required for non-climatology variables"
        init_folder = os.path.join(base_path, start_date.strftime('%Y%m%d'))
        ds = xr.open_dataset(f"{init_folder}/{var}.nc")
    return ds

# Calculate mean temperature as before
def plot_variable(ds, var, forecast_day=0): 
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    assert forecast_day < len(ds['forecast_day']), "forecast_day is out of bounds"
    var_ds = ds[var].sel(forecast_day=ds['forecast_day'][forecast_day])
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 8),
                        subplot_kw={'projection': projection})

    # Add coastlines and borders
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

    # Create filled contour plot
    plot = ax.pcolormesh(var_ds.lon, var_ds.lat, var_ds,
                        transform=projection,
                        shading='auto',
                        cmap='RdYlBu_r')

    # Add colorbar
    plt.colorbar(plot, ax=ax, label=f'{ds[var].units}')

    # Set extent to match your data
    ax.set_extent([
        float(ds.lon.min()),
        float(ds.lon.max()),
        float(ds.lat.min()),
        float(ds.lat.max())
    ])

    # Add gridlines
    ax.gridlines(draw_labels=True)

    # Add title
    start_date = ds['forecast_day'][0].values.astype('datetime64[s]').astype(datetime).strftime('%Y-%m-%d')
    plt.title(f'{forecast_day}d {var} forecast @ {start_date}')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('start_date', nargs='?', help='Start date in YYYYMMDD format')
    args = parser.parse_args()
    start_date = args.start_date
    if start_date is None:
        start_date = get_previous_monday()
    else:
        start_date = datetime.strptime(start_date, '%Y%m%d')
    download_salient_90d_forecasts(start_date)   
    combine_and_save_forecast_stats(start_date)   
    # ds = read_salient_90d_forecasts('solar_radiation', start_date)