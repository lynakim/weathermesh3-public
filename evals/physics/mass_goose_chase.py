from eval_helpers import *
import xarray as xr
import matplotlib.pyplot as plt
Gzarr = xr.open_zarr('gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr')
gc_batch = Gzarr.sel(time='2020-08-27T12:00:00')

gc_batch['air_density'] = calculate_air_density(gc_batch)
gc_batch['height'] = calculate_layer_heights(gc_batch)
gc_batch['volume'] = calculate_cell_volumes(gc_batch)
gc_batch['dry_air_mass'] = gc_batch['air_density'] * gc_batch['volume']

# %%
gc_batch['air_density'].sum(dim=['lat', 'lon', 'level']).values
density_percentage_change = (gc_batch['air_density'].sum(dim=['lat', 'lon', 'level']) - gc_batch['air_density'].sum(dim=['lat', 'lon', 'level'])[0]) / gc_batch['air_density'].sum(dim=['lat', 'lon', 'level'])[0] * 100
plt.plot(density_percentage_change)
plt.title('Percentage Change in Air Density')
plt.xlabel('Time')
plt.ylabel('Percentage Change')
plt.show()
# %%
gc_batch['volume'].sum(dim=['lat', 'lon', 'level']).values
volume_percentage_change = (gc_batch['volume'].sum(dim=['lat', 'lon', 'level']) - gc_batch['volume'].sum(dim=['lat', 'lon', 'level'])[0]) / gc_batch['volume'].sum(dim=['lat', 'lon', 'level'])[0] * 100
plt.plot(volume_percentage_change)
plt.title('Percentage Change in Volume')
plt.xlabel('Time')
plt.ylabel('Percentage Change')
plt.show()
# %%
mass_percentage_change = (gc_batch['dry_air_mass'].sum(dim=['lat', 'lon', 'level']) - gc_batch['dry_air_mass'].sum(dim=['lat', 'lon', 'level'])[0]) / gc_batch['dry_air_mass'].sum(dim=['lat', 'lon', 'level'])[0] * 100
plt.plot(mass_percentage_change)
plt.title('Percentage Change in Dry Air Mass')
plt.xlabel('Time')
plt.ylabel('Percentage Change')
plt.show()
