import requests
import numpy as np
import datetime
import os
import getpass
import sys

OUTPUT_DIR = "/fast/proc/metar/"

if getpass.getuser() == "kai":
    OUTPUT_DIR = "/Users/kai/Desktop/"


def save_data(output_name, response_text, save_raw=True):
    start_time = datetime.datetime.now()
    fields = response_text.split("\n")[0].split(",")
    data = []

    # there's probably a faster way to do this but w/e
    for line in response_text.split("\n")[1:]:
        if not line:
            continue

        point = {}

        values = line.split(",")
        for i in range(len(values)):
            field = fields[i]
            value = values[i]

            if field == 'metar' or field.startswith('skyc') or field == 'wxcodes':
                continue
            elif field == 'station':
                value = value.strip()
            elif field == 'valid' or field == 'peak_wind_time':
                if value == 'M':
                    value = np.nan
                else:
                    value = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M")
                    value = value.replace(tzinfo=datetime.timezone.utc)
                    value = int(value.timestamp())
            elif value in ('M', 'T'):
                value = np.nan
            # else:
            #     try:
            #         value = float(value)
            #     except ValueError as e:
            #         print("Non-numeric value", field, value)
            #         raise e

            point[field] = value

        data.append(point)

    if len(data) == 0:
        print(f"No data for {output_name}")
        return

    dtype = []

    for field in data[0].keys():
        nptype = {
            'station': 'S4',
            'valid': 'int64',
            'peak_wind_time': 'float64'
        }.get(field, 'float32')

        dtype.append((field, nptype))

    structured_array = np.array([tuple(d.values()) for d in data], dtype=dtype)

    np.save(OUTPUT_DIR + output_name, structured_array)

    if save_raw:
        with open(OUTPUT_DIR + output_name.replace('.npy', '.csv'), "w") as f:
            f.write(response_text)

    print(f"Saved {output_name} in {(datetime.datetime.now() - start_time).total_seconds()}s")


def download_metar_on(date, check_existing=True, use_raw=False):
    """
    Downloads data on a given date (UTC)
    use_raw: set to true to if you want to test out reprocessing the raw data, but this will use a lot more disk space
    """

    output_name = f"metar_{date.year}_{date.month}_{date.day}.npy"
    if check_existing and not use_raw and os.path.exists(OUTPUT_DIR + output_name):
        print(f"Data for {date} already exists")
        return

    # https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?data=all&year1=2024&month1=1&day1=1&year2=2024&month2=1&day2=1&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=M&trace=T&direct=no&report_type=3&report_type=4

    params = {
        # 'data': 'all',
        'data': ['station', 'valid', 'lon', 'lat', 'elevation', 'tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'p01i', 'alti', 'mslp', 'vsby', 'gust', 'skyl1', 'skyl2', 'skyl3', 'skyl4', 'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr', 'peak_wind_gust', 'peak_wind_drct', 'peak_wind_time', 'feel', 'snowdepth'],
        'month1': date.month,
        'day1': date.day,
        'year1': date.year,
        'month2': date.month,
        'day2': date.day,
        'year2': date.year,
        'format': 'onlycomma',
        'tz': 'Etc/UTC',
        'latlon': 'yes',
        'elev': 'yes',
        'missing': 'M',
        'trace': 'T',
        'direct': 'no',
        'report_type': ['3', '4']
    }

    if use_raw and os.path.exists(OUTPUT_DIR + output_name.replace('.npy', '.csv')):
        print(f"Reprocessing data on {date}")
        with open(OUTPUT_DIR + output_name.replace('.npy', '.csv'), "r") as f:
            response_text = f.read()
            save_data(output_name, response_text, save_raw=False)
        return

    print(f"Downloading data on {date}")
    start_time = datetime.datetime.now()
    response = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params=params)
    print(f"Downloaded in {(datetime.datetime.now() - start_time).total_seconds()}s")

    if not response.ok:
        print(f"Failed to fetch data on {date}: {response.status_code}")
        return

    save_data(output_name, response.text, save_raw=use_raw)


def debug_metar_save(date):
    output_name = f"metar_{date.year}_{date.month}_{date.day}.npy"
    data = np.load(OUTPUT_DIR + output_name)

    # print out what keys it has and any other metadata
    if data.dtype.names is not None:
        print("Keys available:", data.dtype.names)
    else:
        print("This array does not have keys.")

    print(data.shape)

    # print out the first few entries
    for i in range(min(5, len(data))):
        print(data[i])


def main():
    if '--single' in sys.argv:
        date = datetime.datetime(2024, 1, 1)
        download_metar_on(date)
        debug_metar_save(date)
        return

    # download up until last week (in case there's weirdness with recent stuff not being fully out yet)
    end_date = datetime.datetime.now() - datetime.timedelta(days=7)

    # download last 30 years of data, ish
    for i in range(30*365):
        date = end_date - datetime.timedelta(days=i)
        download_metar_on(date)


if __name__ == "__main__":
    main()
