import numpy as np
from datetime import datetime, timedelta, timezone
import sys
sys.path.append("../../")
from utils import get_dates, log_vars
import pickle
import multiprocessing as mp
import time
import logging
import os
import gc

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

sfc_agg_fns = {
    "201_mx2t": np.max,
    "202_mn2t": np.min,
    }

def daily_avg_npzs(args):
    """Process all files for one date with explicit cleanup"""
    date, var, reprocess = args  # Unpack the arguments tuple
    t0 = time.time()
    var_path = f"{var}" if var == 'f000' else f"extra/{var}"
    try:
        # Check if output already exists
        output_file = f"/fast/proc/era5_daily/{var_path}/{date.strftime('%Y')}/{date.strftime('%Y%m%d')}.npz"
        if not reprocess and os.path.exists(output_file):
            logging.info(f"Skipping {date.strftime('%Y-%m-%d')} because it already exists")
            return date, True, 0

        if var in log_vars:
            with open(f"/fast/consts/normalization.pickle", "rb") as f:
                norm = pickle.load(f)
                mean, std = norm[var][0], np.sqrt(norm[var][1])
        
        sfc_data = []
        pr_sum = None
        n_files = 0
        
        # Process each file and accumulate sums
        for hour in range(24):
            dt = date + timedelta(hours=hour)
            filepath = f"/fast/proc/era5/{var_path}/{dt.strftime('%Y%m')}/{int(dt.timestamp())}.npz"
            if not os.path.exists(filepath):
                filepath = f"/huge/proc/era5/{var_path}/{dt.strftime('%Y%m')}/{int(dt.timestamp())}.npz"
            assert os.path.exists(filepath), f"File {filepath} does not exist"
            try:
                data = np.load(filepath)
                if var != 'f000':
                    data_var = data['x'].copy()
                    if len(data_var.shape) == 3:
                        assert data_var.shape[0] == 1, f"Expected 2d data for {var}, got {data_var.shape} "
                        data_var = data_var[0]
                    if var in log_vars:
                        data_var = data_var * std + mean
                        data_var = np.exp(data_var)
                    sfc_data.append(data_var)
                else:
                    if hour == 0:
                        pr_sum = data['pr'].copy()
                    else:
                        pr_sum += data['pr']
                    sfc_data.append(data['sfc'])
                n_files += 1
                del data
                gc.collect()
                logging.info(f"Loaded file {n_files}/24 for {date.strftime('%Y-%m-%d')}")
            except Exception as e:
                logging.error(f"Error processing {filepath}: {e}")
                continue
        
        if n_files == 0:
            raise ValueError(f"No files processed for {date}")
            
        # Calculate averages
        agg_fn = sfc_agg_fns[var] if var in sfc_agg_fns else np.mean
        sfc_agg = agg_fn(sfc_data, axis=0)
        assert sfc_agg.shape == sfc_data[0].shape, f"sfc_agg shape {sfc_agg.shape} does not match expected {sfc_data[0].shape}"
        if var == 'f000':
            daily_avg = {
                'sfc': sfc_agg,
                'pr': pr_sum / n_files
            }
        else:
            if var in log_vars:
                sfc_agg = (np.log(sfc_agg) - mean) / std
            daily_avg = {
                'x': sfc_agg,
            }
        
        # Save result in year folder
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.savez_compressed(output_file, **daily_avg)
        
        del daily_avg, sfc_data, sfc_agg, pr_sum
        gc.collect()
        
        proc_time = time.time() - t0
        logging.info(f"Completed {date.strftime('%Y-%m-%d')} in {proc_time:.2f}s")
        return date, True, proc_time
        
    except Exception as e:
        logging.error(f"Failed to process {date}: {e}")
        return date, False, time.time() - t0

if __name__ == '__main__':
    # Add summary statistics storage
    summary_stats = {}
    
    # List of variables to process
    # Process in smaller batches
    batch_size = 80
    n_processes = 64 # 50min with 64 procs, 100min for 1y with 40 procs, 166min with 64 procs
    debug_mode = False
    VARS_TO_PROCESS = [
        'f000',  # base
        #'45_tcc', '034_sstk', '168_2d', '179_ttr', '136_tcw', '137_tcwv'
        #'142_lsp', '143_cp', 'logtp', '201_mx2t', '202_mn2t'
    ]

    START_DATE = datetime(1988, 1, 4, tzinfo=timezone.utc)
    END_DATE = datetime(1988, 1, 5, tzinfo=timezone.utc)

    reprocess = True

    if len(sys.argv) > 1:
        try:
            # Parse date from command line argument in format YYYYMMDD
            date_str = sys.argv[1]
            target_date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print("Error: Date must be in format YYYYMMDD (e.g., 20240315)")
            exit()
        # Get all hours for that day
        for var in VARS_TO_PROCESS:
            daily_avg_npzs((target_date, var, reprocess))
        exit()
    else:
        dates = [START_DATE + timedelta(days=x) for x in range((END_DATE - START_DATE).days + 1)]

    # Loop through each variable
    for var in VARS_TO_PROCESS:
        logging.info(f"Starting processing for variable: {var}")
        t0_total = time.time()
        completed_dates = []
        failed_dates = []

        for i in range(0, len(dates), batch_size):
            batch_dates = dates[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{len(dates)//batch_size + 1}")
            
            if debug_mode:
                results = [daily_avg_npzs((date, var, reprocess)) for date in batch_dates]
            else:
                with mp.Pool(n_processes) as pool:
                    results = pool.map(daily_avg_npzs, [(date, var, reprocess) for date in batch_dates])
                
            for date, success, proc_time in results:
                if success:
                    completed_dates.append(date)
                else:
                    failed_dates.append(date)
            
            logging.info(f"Batch complete. Total processed: {len(completed_dates)}/{len(dates)}")
        
        total_time = time.time() - t0_total
        # Store statistics instead of logging immediately
        summary_stats[var] = {
            'total_time': total_time,
            'completed': len(completed_dates),
            'total': len(dates),
            'failed_dates': [d.strftime('%Y-%m-%d') for d in failed_dates]
        }

    # Print final summary for all variables
    logging.info("\n=== PROCESSING SUMMARY ===")
    for var, stats in summary_stats.items():
        logging.info(f"\nVariable: {var}")
        logging.info(f"Total processing time: {stats['total_time']:.2f}s")
        logging.info(f"Successfully processed: {stats['completed']}/{stats['total']}")
        if stats['failed_dates']:
            logging.error(f"Failed dates: {stats['failed_dates']}")
