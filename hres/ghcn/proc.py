import os
import gzip
import pandas as pd
from datetime import datetime
import shutil
from tqdm import tqdm
import pickle
import numpy as np
from calendar import monthrange



base_dir = "raw"
output_dir = "processed"
years = range(1979, 2024)

for year in years:
    # UNZIP
    year_dir = os.path.join(base_dir, f"ghcn-hourly_v1.0.0_d{year}_c20240301.tar.gz")
    with gzip.open(year_dir, 'rb') as f_in:
        with open(year_dir[:-3], 'wb') as f_out:
            f_out.write(f_in.read())
    shutil.unpack_archive(year_dir[:-3], base_dir)
    os.remove(year_dir[:-3])
    year_dir = os.path.join(base_dir, f"{year}")

    # create {output_dir}/{year}/ if it doesn't exist
    if not os.path.exists(f"{output_dir}/{year}"):
        os.makedirs(f"{output_dir}/{year}")



    # CREATE STATION PICKLE FILE IF IT DOESN'T EXIST, OTHERWISE LOAD IT
    try:
        with open('stations.pickle', 'rb') as handle:
            stations = pickle.load(handle)
            stations = {stations[i]: i for i in range(len(stations))}
    except FileNotFoundError:
        stations = {}


    data_dictionary = {}



    new_entries = []
    # FOR EACH STATION:
    station_file_list = os.listdir(year_dir)[:]
    for station_file in tqdm(station_file_list):
        # LOAD DATA, METADATA WITH PANDAS
        filepath = os.path.join(year_dir, station_file)
        df = pd.read_csv(filepath, delimiter='|', low_memory=False)
        station_ID = df['Station_ID'][0]
        station_latlon = (float(df['Latitude'][0]), float(df['Longitude'][0]))

        # CHECK FOR STATION IN stations
        if station_latlon not in stations:
            new_entries.append(station_latlon)
            stations[station_latlon] = len(stations)

        station_index = stations[station_latlon]

        # Create a boolean mask for valid days
        df['num_days'] = df.apply(lambda row: monthrange(row['Year'], row['Month'])[1], axis=1)
        valid_days_mask = df['Day'] <= df['num_days']
        df = df[valid_days_mask]
        df.reset_index(drop=True, inplace=True)


        #ensure that all of df['wind_direction'] is either a number or a nan
        df['wind_direction'] = pd.to_numeric(df['wind_direction'], errors='coerce')
        wind_direction_rad = np.deg2rad(df['wind_direction'])
        df['u_wind'] = -df['wind_speed'] * np.sin(wind_direction_rad)
        df['v_wind'] = -df['wind_speed'] * np.cos(wind_direction_rad)


        #WEIRD SPIKE QC
        #PROCESS HERE
    

        years, months, days, hours, minutes = df['Year'], df['Month'], df['Day'], df['Hour'], df['Minute']
        temperatures = df['temperature']
        dewpoints = df['dew_point_temperature']
        sea_level_pressures = df['sea_level_pressure']
        wind_speeds = df['wind_speed']
        wind_directions = df['wind_direction']
        u_winds = df['u_wind']
        v_winds = df['v_wind']
        cloud_ceiling_heights = df['sky_cover_baseht_1']
        visibilities = df['visibility']
        precipitations = df['precipitation']


        for i in range(len(df)):
            # DERIVE OBSERVATION HOUR
            hour_time = datetime(years[i], months[i], days[i], hours[i])
            observation_time = datetime(years[i], months[i], days[i], hours[i], minutes[i])
            timestep = (observation_time - hour_time).total_seconds()

            temperature = temperatures[i]
            dewpoint = dewpoints[i]
            sea_level_pressure = sea_level_pressures[i]
            u_wind = u_winds[i]
            v_wind = v_winds[i]
            cloud_ceiling_height = cloud_ceiling_heights[i]
            visibility = visibilities[i]
            precipitation = precipitations[i]

            if not -122 < temperature < 50:
                temperature = np.nan
            if not -122 < dewpoint < 50:
                dewpoint = np.nan
            if not -100 < u_wind < 100:
                u_wind = np.nan
            if not -100 < v_wind < 100:
                v_wind = np.nan
            if not 850 < sea_level_pressure < 1100:
                sea_level_pressure = np.nan
            if not precipitation <= 250:
                precipitation = np.nan
            if not visibility <= 200:
                visibility = np.nan

            row_data = [
                timestep,
                station_index,
                sea_level_pressure,
                temperature,
                dewpoint,
                u_wind,
                v_wind,
                cloud_ceiling_height,
                visibility,
                precipitation,
            ]

            #if every value in row_data is nan, continue
            if all(np.isnan(row_data[2:])):
                continue

            datestr = observation_time.strftime('%Y%m%d%H')
            if datestr not in data_dictionary:
                data_dictionary[datestr] = [row_data]
            else:
                data_dictionary[datestr].append(row_data)

    for datestr, data in data_dictionary.items():
        data = np.array(data, dtype=np.float32)
        np.save(f"{output_dir}/{year}/{datestr}.npy", data)

    if new_entries:
        print(f"Added {len(new_entries)} new stations")
        with open('stations.pickle', 'wb') as handle:
            pickle.dump(list(stations), handle)
