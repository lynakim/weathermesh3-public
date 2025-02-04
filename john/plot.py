import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#plt.style.use('wb.mplstyle'); print("🎨🎨🎨 FYI, setting WindBorne style for plotting")

def get_map_axes():
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    plt.tight_layout()
    return ax

