from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
from utils import get_raw_nc_filename, get_dataset
import numpy as np

# check outlier max norm values for 100v

# Set up logging
logging.basicConfig(
    filename='zeros_100v.log',
    level=logging.INFO,
    format='%(message)s'
)

def process_file(date):
    try:
        fn = get_raw_nc_filename("247_100v", date)
        data = get_dataset(fn)
        
        # Convert to numpy array if it isn't already
        if hasattr(data, 'values'):
            data = data.values
            
        # Calculate statistics
        min_sum = np.min(np.abs(np.sum(data, axis=(1,2,3))))
        
        # Log results
        log_msg = f"{fn},{min_sum}"
        logging.info(log_msg)
        
        # Return if we want to collect results
        return fn, min_sum
        
    except Exception as e:
        logging.error(f"Error processing {date}: {str(e)}")
        return None

def main():
    # Generate dates
    dates = [
        datetime(year, month, 1) 
        for year in range(1980, 2024) 
        for month in range(1, 13)
    ]
    
    # Create output directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Process files in parallel
    # Adjust max_workers based on your CPU cores (typically num_cores - 1)
    with ProcessPoolExecutor(max_workers=15) as executor:
        # Submit all tasks and get futures
        futures = [executor.submit(process_file, date) for date in dates]
        
        # Collect results as they complete
        results = []
        for future in futures:
            result = future.result()
            if result:
                results.append(result)

    # Optional: Save summary statistics
    with open('wind_summary.csv', 'w') as f:
        f.write("date,filename,max_speed\n")
        for result in results:
            if result:
                fn, max_val = result
                f.write(f"{fn},{max_val:.2f}\n")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")