import numpy as np
import os
import sys
sys.path.append('/fast/wbhaoxing/windborne')
from meteo.tools.process_dataset import download_s3_file
np.set_printoptions(precision=4, suppress=True)

from datetime import datetime, timezone, timedelta


WM_folder = "WeatherMesh-backtest"
WM_hash = "Qfiz"

date_start = datetime(2024, 9, 21)
date_end = datetime(2024, 9, 28)
date_range = [date_start + timedelta(days=i) for i in range((date_end - date_start).days + 1)]
print(f"date_range: {date_range}")

fhs = list(range(0, 241, 6))

for date in date_range:
    date_str = date.strftime("%Y%m%d") + "00"
    for fh in fhs:
        os.makedirs(f"/huge/deep/realtime/outputs/{WM_folder}/{date_str}/det", exist_ok=True)
        download_s3_file("wb-dlnwp", f"{WM_folder}/{date_str}/det/{fh}.{WM_hash}.npy",f"/huge/deep/realtime/outputs/{WM_folder}/{date_str}/det/{fh}.{WM_hash}.npy")