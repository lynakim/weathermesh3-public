from collections import defaultdict
import json
import os
import pygrib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys
sys.path.append('/fast/haoxing/windborne')
from meteo.tools.process_dataset import download_s3_file
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt

##########################################################################

# This script generates the WM vs HRES 2m temperature table for ~any list 
# of cities. To use this script, set the variables in Steps 1-6 in the
# next section, and then run the script.

# Contact Haoxing for any questions or issues.

##########################################################################

# Step 1. Replace with cities of interest

# Need to be cities with airports with ICAO codes that report METARs
# Copilot is quite good at giving you the ICAO code of an airport if you just write the city name
# This sets the order and display names of the cities in the table as well
CITIES = {
    # "Atlanta, GA": b"ATL",
    # "Boston, MA": b"BOS",
    # "Chicago, IL": b"ORD",
    # "Houston, TX": b"IAH",
    # "Miami, FL": b"MIA",
    # "New York, NY": b"LGA",
    # "San Jose, CA": b"SJC",
    # "Barcelona, Spain": b"LEBL",
    # "Beijing, China": b"ZBAA",
    # "London, UK": b"EGLL",
    # "Mumbai, India": b"VABB",
    # "Nairobi, Kenya": b"HKJK",
    # "Sao Paolo, Brazil": b"SBGR",
    # "Seoul, South Korea": b"RKSI",
    # "Toronto, Canada": b"CYYZ",
    "São Paulo": b"SBGR",
    "Rio de Janeiro": b"SBGL",
    "Belo Horizonte": b"SBCF",
    "Brasília": b"SBBR",
    "Salvador": b"SBSV",
    "Fortaleza": b"SBFZ",
    "Recife": b"SBRF",
    "Manaus": b"SBEG",
    "Curitiba": b"SBCT",
    "Belem": b"SBBE",
}

# Step 2. Set the evaluation period

EVAL_START = datetime(2024, 7, 1, tzinfo=timezone.utc)
EVAL_END = datetime(2024, 8, 31, tzinfo=timezone.utc)

# Step 3. Set forecast hours for WeatherMesh and HRES

# HRES is available up to 10 days
WM_FORECAST_HOURS = list(range(24, 24*14+1, 24))
HRES_FORECAST_HOURS = list(range(24, 24*10+1, 24))

# Step 4. Set the sources for forecast data

# WM
# note: for WM you may need to run backtests on your dates.
# Ping Haoxing if you need help with this.
WM_LOCAL_PATH = "/huge/deep/realtime/outputs/WeatherMesh-backtest"
WM_S3_BUCKET = "wb-dlnwp"
WM_S3_PATH = "WeatherMesh-backtest"
WM_VARIANT = "vGM0"
# HRES
# note: this script gets HRES forecasts from our own wx-archive, which
# has forecasts from March 2024 onwards. If you need forecasts from before,
# you may need to make a MARS request. Ping Haoxing if you need help with this.
HRES_LOCAL_PATH = "/huge/proc/weather-archive/ecmwf"

# Step 5. Set raw result output paths

WM_ERRORS_PATH = "/fast/haoxing/deep/evals/cities/city_wm_errors_brazil_summer.json"
HRES_ERRORS_PATH = "/fast/haoxing/deep/evals/cities/city_hres_errors_brazil_summer.json"

# Step 6. Set table parameters and output path

# if you want to mess with the actual table formatting, go to the end of this script
TITLE = "2m Temperature Forecasts Verified Against\nJuly-August 2024 METAR Observations"
OUTPUT_PATH = "/fast/haoxing/deep/evals/cities/city_wm_vs_hres_brazil_summer.png"

##########################################################################

def get_closest_utc_whole_hour(timestamp: int) -> datetime:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if dt.minute < 30:
        return dt.replace(minute=0, second=0)
    else:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0)

meta = json.load(open("/huge/deep/realtime/outputs/yamahabachelor/meta.vGM0.json", "r"))
wm_lats, wm_lons = meta["lats"], meta["lons"]
wm_lons = wm_lons[720:] + wm_lons[:720]

# gather city observations
print("Gathering city observations")
city_obs = defaultdict(list)

day = EVAL_START
while day <= EVAL_END + timedelta(days=20): # need to add ~20 days since forecast out to 14 days
    metar = np.load(f"/fast/proc/metar/metar_{day.year}_{day.month}_{day.day}.npy")
    for entry in metar:
        if entry[0] in CITIES.values():
            utc_dt = get_closest_utc_whole_hour(entry[1])
            city_obs[utc_dt.timestamp()].append(entry)
    day += timedelta(days=1)

# compute linearly interpolated WM and HRES errors
print("Computing WeatherMesh & HRES errors")
wm_errors, hres_errors = defaultdict(list), defaultdict(list)

day = EVAL_START
while day <= EVAL_END:
    init_datestr = day.strftime("%Y%m%d") + "00"
    for fh in WM_FORECAST_HOURS:
        obs_date = day + timedelta(hours=fh)

        # WM
        print(f"Processing WM for init {init_datestr} forecast hour {fh}")
        os.makedirs(f"{WM_LOCAL_PATH}/{init_datestr}/det", exist_ok=True)
        download_s3_file(WM_S3_BUCKET, f"{WM_S3_PATH}/{init_datestr}/det/{fh}.{WM_VARIANT}.npy", f"{WM_LOCAL_PATH}/{init_datestr}/det/{fh}.{WM_VARIANT}.npy")
        wm_out = np.load(f"{WM_LOCAL_PATH}/{init_datestr}/det/{fh}.{WM_VARIANT}.npy")
        wm_out = wm_out[:,:,-5]
        wm_out = np.concatenate([wm_out[:,720:], wm_out[:,:720]], axis=1)
        wm_tempk_interp = RegularGridInterpolator((wm_lats, wm_lons), wm_out, method="linear")

        if fh in HRES_FORECAST_HOURS:
            # HRES
            print(f"Processing HRES for init {init_datestr} forecast hour {fh}")
            os.makedirs(f"{HRES_LOCAL_PATH}/{init_datestr[:-2]}/00z/ifs/0p25/oper", exist_ok=True)
            grbs = pygrib.open(f"{HRES_LOCAL_PATH}/{init_datestr[:-2]}/00z/ifs/0p25/oper/{init_datestr}0000-{fh}h-oper-fc.grib2")
            hres_out = grbs.select(name="2 metre temperature")[0].values[:720,:]
            hres_tempk_interp = RegularGridInterpolator((wm_lats, wm_lons), hres_out, method="linear")

        for entry in city_obs[obs_date.timestamp()]:
            lon, lat, tempf = entry[2], entry[3], entry[5]
            tempk = (tempf - 32) * 5 / 9 + 273.15
            tempk_wm = wm_tempk_interp([lat, lon])[0]
            wm_errors[entry[0].decode()].append((init_datestr, fh, float(tempk), float(tempk_wm), float(tempk - tempk_wm)))
            if fh in HRES_FORECAST_HOURS:
                tempk_hres = hres_tempk_interp([lat, lon])[0]
                hres_errors[entry[0].decode()].append((init_datestr, fh, float(tempk), float(tempk_hres), float(tempk - tempk_hres)))
        
    # save every day
    with open(WM_ERRORS_PATH, "w") as f:
        json.dump(wm_errors, f)
    with open(HRES_ERRORS_PATH, "w") as f:
        json.dump(hres_errors, f)
    
    day += timedelta(days=1)

print("Calculating RMSE and making table")

def dedup_and_calculate_rmse(errors: dict, forecast_hours: list) -> dict:
    # dedup is needed because there appear to be some duplicate observations
    # I don't think the effect is super significant but it's good to be safe
    errors_dedup = defaultdict(list)
    for city, errors in errors.items():
        error_tuples = [tuple(error) for error in errors]
        error_tuples = list(set(error_tuples))
        print(f"{city}: {len(errors)} -> {len(error_tuples)}, number of duplicates: {len(errors) - len(error_tuples)}")
        errors_dedup[city] = error_tuples

    errors_rmse_dedup = defaultdict(list)
    for city, errors in errors_dedup.items():
        for fh in forecast_hours:
            errors_fh = [error[4] for error in errors if error[1] == fh and not np.isnan(error[4])]
            rmse = np.sqrt(np.mean(np.square(errors_fh)))
            errors_rmse_dedup[city].append(float(rmse))
    return errors_rmse_dedup

print("Dedup and calculate RMSE for WM...")
wm_rmse = dedup_and_calculate_rmse(wm_errors, WM_FORECAST_HOURS)
print("Dedup and calculate RMSE for HRES...")
hres_rmse = dedup_and_calculate_rmse(hres_errors, HRES_FORECAST_HOURS)

# remove cities with no data
wm_rmse = {city: rmse for city, rmse in wm_rmse.items() if len(rmse) > 0}
hres_rmse = {city: rmse for city, rmse in hres_rmse.items() if len(rmse) > 0}

# make table
row_labels = list(CITIES.keys())
hres_rmses = np.array([hres_rmse[CITIES[city].decode()] for city in row_labels])
wm_rmses = np.array([wm_rmse[CITIES[city].decode()] for city in row_labels])
col_labels = [1,3,5,7,10,"14*"]
hres_rmses = hres_rmses[:,np.array([0,2,4,6,9,9])] * 9/5
wm_rmses = wm_rmses[:,np.array([0,2,4,6,9,13])] * 9/5

fig, ax = plt.subplots(figsize=(8, 14), dpi=300)

ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticklabels(row_labels)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(row_labels)):
    for j in range(len(col_labels)):
        text = ax.text(j, i, f"{wm_rmses[i, j]:.1f} | {hres_rmses[i, j]:.1f}", ha="center", va="center", color="black")

cax = ax.imshow(wm_rmses - hres_rmses, cmap='RdBu_r', alpha=0.8, vmin=-4, vmax=4, aspect=0.5)
cbar = fig.colorbar(cax, orientation='horizontal', pad=0.1, fraction=0.02)
cbar.ax.set_xticks([])
cbar.set_label("WeatherMesh 2.0 better                                            IFS better")


ax.set_title(f"{TITLE}\n", fontweight="bold")
secondary_x = ax.secondary_xaxis('top')
secondary_x.set_xlabel("WeatherMesh 2.0 RMSE (F) | ECMWF IFS RMSE (F)")
secondary_x.set_xticks([])
ax.set_xlabel("Forecast Lead Time (Days)")

fig.text(0.9, 0.06, "* WeatherMesh 14-day RMSE compared with IFS 10-day RMSE", ha='right', fontsize=9)

# save
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"Table saved to {OUTPUT_PATH}")