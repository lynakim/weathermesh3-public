import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
import requests
import gzip
import datetime
import pandas as pd
import numpy as np
import pickle
import os
from calendar import monthrange

INGEST = True
DOWNLOAD_DIR = '/Users/clamalo/downloads'
observation_year = 2023

# if the year directory does not exist, create it
if not os.path.exists(f'{DOWNLOAD_DIR}/raw/{observation_year:04d}'):
    os.makedirs(f'{DOWNLOAD_DIR}/raw/{observation_year:04d}')
if not os.path.exists(f'{DOWNLOAD_DIR}/processed/{observation_year:04d}'):
    os.makedirs(f'{DOWNLOAD_DIR}/processed/{observation_year:04d}')

for observation_month in range(1, 13):
    num_days = monthrange(observation_year, observation_month)[1]
    for observation_day in range(1, num_days+1):

        for observation_hour in range(0, 24):
            observation_minute = 0
            observation_datetime = datetime.datetime(observation_year, observation_month, observation_day, observation_hour, observation_minute)
            obs_datetime_np = np.datetime64(observation_datetime)

            if INGEST:
                url = f"https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/{observation_year:04d}/{observation_month:02d}/{observation_day:02d}/LDAD/mesonet/netCDF/{observation_year:04d}{observation_month:02d}{observation_day:02d}_{observation_hour:02d}00.gz"
                print(url)

                response = requests.get(url, stream=True)

                # Total size in bytes.
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 # 1 Kibibyte

                t = tqdm(total=total_size, unit='iB', unit_scale=True)

                with open(f'{DOWNLOAD_DIR}/raw/{observation_year:04d}/{observation_month:02d}{observation_day:02d}{observation_hour:02d}.gz', 'wb') as f:
                    for data in response.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                t.close()

            # Unzip the file
            with gzip.open(f'{DOWNLOAD_DIR}/raw/{observation_year:04d}/{observation_month:02d}{observation_day:02d}{observation_hour:02d}.gz', 'rb') as f_in:
                with open(f'{DOWNLOAD_DIR}/raw/{observation_year:04d}/{observation_month:02d}{observation_day:02d}{observation_hour:02d}.nc', 'wb') as f_out:
                    f_out.write(f_in.read())

            ds = xr.load_dataset(f'{DOWNLOAD_DIR}/raw/{observation_year:04d}/{observation_month:02d}{observation_day:02d}{observation_hour:02d}.nc')

            os.remove(f'{DOWNLOAD_DIR}/raw/{observation_year:04d}/{observation_month:02d}{observation_day:02d}{observation_hour:02d}.nc')

            # If ds doesn't have skyCovLayerBase, add it with nan values
            if 'skyCovLayerBase' not in ds:
                ds['skyCovLayerBase'] = np.nan

            # Select only the needed variables
            ds = ds[['stationName', 'stationType', 'observationTime', 'latitude', 'longitude', 'elevation', 'temperature', 'dewpoint', 'seaLevelPressure', 'windDir', 'windSpeed', 'skyCovLayerBase', 'visibility']]

            ds['observationTime'] = ds['observationTime'].astype('datetime64[ns]')
            time_diff = abs(ds['observationTime'] - obs_datetime_np)
            master_ds = ds.where(time_diff <= np.timedelta64(1, 'h'), drop=True)

            wind_dir_rad = np.deg2rad(master_ds.windDir)
            # Calculate the u and v components of the wind
            u = -master_ds.windSpeed * np.sin(wind_dir_rad)
            v = -master_ds.windSpeed * np.cos(wind_dir_rad)
            # Add these components back to the dataset
            master_ds['u_wind'] = u
            master_ds['v_wind'] = v

            # Load existing station coordinates
            try:
                with open('station_latlon.pickle', 'rb') as handle:
                    station_latlon = set(pickle.load(handle))
            except FileNotFoundError:
                station_latlon = set()

            # Check and add new station coordinates
            new_entries = []
            for station in tqdm(range(len(master_ds.stationName))):
                station_info = (master_ds.latitude.values[station], master_ds.longitude.values[station])
                if station_info not in station_latlon:
                    new_entries.append(station_info)
                    station_latlon.add(station_info)

            # Save the updated station coordinates if new entries are found
            if new_entries:
                print(f"Added {len(new_entries)} new stations")
                with open('station_latlon.pickle', 'wb') as handle:
                    pickle.dump(list(station_latlon), handle)

            # Make dict of stations (key is lat, lon, value is index)
            station_latlon = list(station_latlon)
            station_latlon = {station_latlon[i]: i for i in range(len(station_latlon))}

            # Dictionary to keep track of the latest observation time for each station
            latest_observation_time = {}

            station_data = set()

            for station in tqdm(range(len(master_ds.stationName))):
                obs_timestamp = pd.to_datetime(master_ds.observationTime.values[station]).timestamp()
                station_index = station_latlon[(master_ds.latitude.values[station], master_ds.longitude.values[station])]

                # Check for duplicates efficiently
                if station_index in latest_observation_time and abs(obs_timestamp - latest_observation_time[station_index]) <= 840:
                    continue

                # Update the latest observation time for the station
                latest_observation_time[station_index] = obs_timestamp

                data = [
                    obs_timestamp,
                    station_index,
                    master_ds.temperature.values[station],
                    master_ds.dewpoint.values[station],
                    master_ds.seaLevelPressure.values[station],
                    master_ds.u_wind.values[station],
                    master_ds.v_wind.values[station],
                    np.nan,
                    master_ds.visibility.values[station]
                ]

                station_data.add(tuple(data))

            # Print length of station_data
            print(f"Added {len(station_data)} new entries to {observation_year:04d}/{observation_month:02d}{observation_day:02d}{observation_hour:02d}.npy")

            # Save the updated station DATA if new entries are found
            with open(f'{DOWNLOAD_DIR}/processed/{observation_year:04d}/{observation_month:02d}{observation_day:02d}{observation_hour:02d}.npy', 'wb') as handle:
                np.save(handle, np.array(list(station_data)))  # save