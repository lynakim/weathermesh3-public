import sys
sys.path.append('.')

from datetime import datetime, timezone
import time
import numpy as np
import os
import argparse
import boto3
from botocore.exceptions import ClientError

from utils import *
from data import *
from eval import *

from evals.evaluate_error import *
from evals.package_neo import *
from run_rt_TC import RunTCModel

# Calculates the amount of data required to generate this forecast
# Assumes 300MB per valid at time:
#   ((720 * 1440 * ~140 pixels) * (2 bytes / pixel (fp16)) = ~ 300 MB)
def _forecast_data_required(forecast_dates, hour_separation, days_per_forecast):
    forecast_size = 300
    num_forecasts = len(forecast_dates)
    num_forecasts_at_time = int(days_per_forecast * 24 / hour_separation)
    return forecast_size * num_forecasts * num_forecasts_at_time / (1e6) # Return in TB

def _gather_forecast_dates(forecast_start, forecast_end, hour_separation):
    dates = []
    current_date = forecast_start
    while current_date <= forecast_end:
        dates.append(current_date)
        current_date += timedelta(hours=hour_separation)
    print(GREEN(f"Generated {len(dates)} forecast dates"))
    return dates

# OLD MODEL NAMES
# TCregional
# TCregionalio
# TCfullforce
# TCvarweight 

def main():
    # Runs like:
    # python3 realtime/run_TC/run_cumulative_TC.py [MODEL_NAME] YYYYMMDDHH YYYYMMDDHH [GPU] --hour_separation [HOUR SEPARATION] --days_per_forecast [DAYS PER FORECAST]
    # python3 realtime/run_TC/run_cumulative_TC.py bachelor 2024061300 2024083100 --gpu 1 --hour_separation 6 --days_per_forecast 5
    # python3 realtime/run_TC/run_cumulative_TC.py operational_stackedconvplus 2024061300 2024083100 --gpu 0 --hour_separation 6 --days_per_forecast 5
    # 
    # POTENTIAL MODEL NAMES
    # bachelor (must be called with bachelor flag)
    # operational_stackedconvplus 
    print(f"Started process {os.getpid()} for run_rt_TC", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help="Model name to run")
    parser.add_argument('forecast_start', type=str, help='Forecast start time in YYYYMMDDHH format')
    parser.add_argument('forecast_end', type=str, help='Forecast end time in YYYYMMDDHH format')
    parser.add_argument('--gpu', type=int, required=True, help='GPU to run on')
    parser.add_argument('--hour_separation', type=int, default=6, help='Hours between forecast (dt)')
    parser.add_argument('--days_per_forecast', type=int, default=5, help='Days forecasting out to for a particular forecast date')
    args = parser.parse_args()

    model_name = args.model_name
    forecast_start = datetime.strptime(args.forecast_start, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    forecast_end = datetime.strptime(args.forecast_end, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    assert isinstance(model_name, str), RED(f"Model name must be a string, got {model_name}")
    assert isinstance(forecast_start, datetime), RED(f"Forecast start must be a datetime object, got {forecast_start}")
    assert isinstance(forecast_end, datetime), RED(f"Forecast end must be a datetime object, got {forecast_end}")
    
    gpu = args.gpu
    hour_separation = args.hour_separation
    days_per_forecast = args.days_per_forecast
    
    forecast_dates = _gather_forecast_dates(forecast_start, forecast_end, hour_separation)
    
    if model_name == 'bachelor':
        def list_available_subsets(s3_client, bucket, date):
            prefix = f"viz/WeatherMesh/tcs/{date.strftime('%Y%m%d%H')}_tcs"
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix
                )
                
                files = []
                for obj in response.get('Contents', []):
                    filename = obj['Key']
                    if filename.endswith('.json'):
                        subset = filename.split('_tcs.')[-1].replace('.json', '')
                        files.append(subset)
                        
                return files
            except ClientError as e:
                print(f"Error listing objects: {str(e)}")
                return []
            
        s3_client = boto3.client('s3')
        s3_bucket = 'wb-dlnwp'
        for date in forecast_dates:
            s3_location = f"viz/WeatherMesh/tcs/{date.strftime('%Y%m%d%H')}_tcs" # .Qfiz.json 
            
            subsets = list_available_subsets(s3_client, s3_bucket, date)
            assert len(subsets) == 1, RED(f"Expected 1 subset for {date}, got {len(subsets)}")
            subset = subsets[0]
            s3_location += f".{subset}.json"
            
            local_path = f"/huge/users/jack/eval_cumulative/bachelor/{date.strftime('%Y%m%d%H')}_tcs.json"
            try:
                s3_client.download_file(s3_bucket, s3_location, local_path)
                print(GREEN(f"Successfully downloaded bachelor data for {date}"))
            except ClientError as e:
                print(RED(f"Failed to download bachelor data for {date} at {s3_location}: {str(e)}"))
                raise Exception("Failed to download bachelor data")
        print(GREEN(f"Successfully downloaded all bachelor data! Exiting..."))
        return 
    
    data_required = _forecast_data_required(forecast_dates, hour_separation, days_per_forecast)
    assert data_required < 3, RED(f"Data required is more than 3 TB ({data_required} TB required), please reduce number of weeks or hour separation")
    print(ORANGE(f"Data required for this forecasting: {data_required}"))
    
    for date in forecast_dates:
        dts = list(range(0, (days_per_forecast * 24) + 1, hour_separation))
        print("--------------------------------------", flush=True)
        print(f"Running model {model_name} at {date}", flush=True)
        start = time.time()
        model = RunTCModel(model_name, dts, date, gpu_id=gpu, output_base_path='/huge/users/jack/eval_cumulative/')
        model.run()
        print(f"Finished running model {model_name} for {date}, took {time.time() - start} seconds", flush=True)
        print("--------------------------------------", flush=True)

if __name__ == '__main__':
    main()