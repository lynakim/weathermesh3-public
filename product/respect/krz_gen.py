# %%
import pandas
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime

actual = pandas.read_parquet('data/de_actual_data.parquet', engine='pyarrow')
daily_avg = pandas.read_parquet('data/de_forecasts_daily_avg.parquet', engine='pyarrow')
spv_ens = pandas.read_parquet('data/de_forecasts_spv_ens.parquet', engine='pyarrow')
wind_ens = pandas.read_parquet('data/de_forecasts_wind_ens.parquet', engine='pyarrow')

def plot_preds_at(date, thing='wind'):
    global actual,wind_ens,spv_ens
    value = pd.to_datetime(date)
    pred = wind_ens if thing == 'wind' else spv_ens
    rows = pred[pred['publication_date'] == value]
    merged_df = actual.join(rows, how='inner', lsuffix='_actual', rsuffix='_pred')
    merged_df.plot(y=['Avg', thing], figsize=(20, 10),label=['predicted', 'actual '+thing])



plot_preds_at('2016-3-2T23:00:00.000Z', 'solar')


# %%

plt.figure(figsize=(20, 10))
plt.plot(actual)
plt.grid()
plt.legend(actual.columns)
plt.ylabel('Power (MW)')
# %%
