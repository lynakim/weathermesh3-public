# dataset page https://apps.ecmwf.int/datasets/data/s2s/ - check request from here
# scriptexamples https://confluence.ecmwf.int/display/WEBAPI/Python+S2S+examples
# TODO: get other recent s2s models in this dataset 

import os
import pygrib
from ecmwfapi import ECMWFDataServer
from datetime import datetime, timedelta
server = ECMWFDataServer()


def get_month_init_dates(year_month):
    # forecasts done on mon and thurs
    date = datetime(int(year_month[:4]), int(year_month[4:6]), 1)
    mon_thu = []
    while date.month == int(year_month[4:6]):
        if date.weekday() == 0 or date.weekday() == 3:
            mon_thu.append(date.strftime("%Y-%m-%d"))
        date += timedelta(days=1)
    dates = '/'.join(mon_thu)
    return dates

def get_ecmwf_data(model, dates, output_file):
    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": dates,
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "origin": model,
        # SST, 2mt, 2m dewpoint, tcc
        "param": "31/33/34/59/136/167/168/235/228032/228086/228087/228095/228096/228141/228164",
        # i think first half of 2015 is available but with slightly different time steps, haven't gotten it yet
        "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104",
        "stream": "enfo",
        "time": "00:00:00",
        "type": "cf",
        "target": output_file
    })

# call this to get all ecmwf grib files
def get_ecmwf_grib(model="ecmf"):
    year_months = [f"{y}{m:02d}" for y in range(2016, 2025) for m in range(1, 13)]
    #year_months  = ['201512', '201511', '201510', '201509', '201508', '201507', '201506', '201505', '201504', '201503', '201502', '201501']
    for ym in year_months:
        dates = get_month_init_dates(ym)
        output_file = f"/huge/proc/s2s/{model}/rt/daily/{ym}.grib"
        if os.path.exists(output_file):
            print(f"Skipping {ym} because it already exists at {output_file}")
            continue
        print(f"Getting data for {ym} with dates {dates} to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        get_ecmwf_data(model, dates, output_file)

def process_grib_to_npz(grib_file, npz_file):
    grbs = pygrib.open(grib_file)
    for grb in grbs:
        print(grb.name, grb.level, grb.values.shape)
