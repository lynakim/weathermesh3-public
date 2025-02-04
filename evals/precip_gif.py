# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta, timezone
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys

sys.path.append('/fast/wbhaoxing/windborne')
from meteo.tools.process_dataset import download_s3_file
# %%
cumulative_precip = np.zeros((720, 1440))
# %%
def plot_us(ax, title, lons, lats, data, vmin=5, vmax=45):
    """Plot temperature data on a map of the US."""
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':')
    ax.set_extent([-100.83, -61.8, 21.1, 52.2])
    ax.set_title(title)
    mesh = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('hourly total precip (cm)')

def visualize_precip(init_date_str: str, forecast_hour: int):
    if not os.path.exists(f"/huge/deep/realtime/outputs/WeatherMesh/{init_date_str}/det/{forecast_hour}.vGM0.npy"):
        os.makedirs(f"/huge/deep/realtime/outputs/WeatherMesh/{init_date_str}/det", exist_ok=True)
    download_s3_file("wb-dlnwp", f"WeatherMesh/{init_date_str}/det/{forecast_hour}.vGM0.npy", f"/huge/deep/realtime/outputs/WeatherMesh/{init_date_str}/det/{forecast_hour}.vGM0.npy")
    wm_out = np.load(f"/huge/deep/realtime/outputs/WeatherMesh/{init_date_str}/det/{forecast_hour}.vGM0.npy")
    meta = json.load(open("/huge/deep/realtime/outputs/yamahabachelor/meta.vGM0.json", "r"))
    lons = meta["lons"]
    lats = meta["lats"]
    # precip = wm_out[:,:,-3]
    # prior to 9/27 or so the precip values were not normalized
    precip = np.exp(wm_out[:,:,-3] * np.sqrt(9.2504) - 12.4496) * 100
    global cumulative_precip
    cumulative_precip += precip
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
    utc_time = datetime.strptime(init_date_str, "%Y%m%d%H") + timedelta(hours=forecast_hour)
    EDT = timezone(timedelta(hours=-4))
    eastern_time = utc_time.astimezone(EDT).strftime("%Y-%m-%d %H:%M")
    plot_us(ax, f"forecast init time: 2024092512z, local time: {eastern_time}", lons, lats, precip, 0, 1)
    plt.savefig(f"/fast/wbhaoxing/helene_precip/{init_date_str}_{forecast_hour}.png")
    plt.show()
    plt.close()

# %%
for fh in range(2, 72, 1):
    visualize_precip("2024092512", fh)
# %%
# make GIF
def make_gif(init_date_str: str):
    images = []
    for fh in range(2, 72, 2):
        images.append(Image.open(f"/fast/wbhaoxing/helene_precip/{init_date_str}_{fh}.png"))
    images[0].save(f"/fast/wbhaoxing/helene_precip/{init_date_str}.gif", save_all=True, append_images=images[1:], loop=0, duration=200)

make_gif("2024092512")
# %%
meta = json.load(open("/huge/deep/realtime/outputs/yamahabachelor/meta.vGM0.json", "r"))
lons = meta["lons"]
lats = meta["lats"]
fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200, subplot_kw={'projection': ccrs.PlateCarree()})
plot_us(ax, "cumulative precip", lons, lats, cumulative_precip, 0, 30)