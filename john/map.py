import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_globe_axis():
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.set_global()
    plt.tight_layout()
    tr = ccrs.PlateCarree()    
    return ax, tr



ax,tr = get_globe_axis()
ax.plot([0,1],[0,1],'r*',transform=tr)
plt.savefig('ohp.png')