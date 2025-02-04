

import mapper 



# %%

import pandas
wind = pandas.read_parquet('data/wind_farms_de.parquet', engine='pyarrow')
solar = pandas.read_parquet('data/solar_farms_de.parquet', engine='pyarrow')
actual = pandas.read_parquet('data/de_actual_data.parquet', engine='pyarrow')
print(actual)


# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([4, 16, 46, 56], crs=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.BORDERS)
ax.coastlines()
plt.scatter(solar['longitude'], solar['latitude'],label='solar')
plt.scatter(wind['longitude'], wind['latitude'],label='wind')
plt.legend()


plt.savefig('ohp.png')


# %%


fig = plt.figure(figsize=(8, 8))

tn = 'gross_performance (kWh)'
plt.hist(solar[solar[tn] < 1e2][tn],bins=10,)
plt.xlabel(tn)
plt.ylabel('Count')
# %%
