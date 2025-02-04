import os
import numpy as np
from tqdm import tqdm

INPUT_DIR = 'processed'
OUTPUT_DIR = 'processed_npz'

# List every year folder
years = sorted([int(x) for x in os.listdir(INPUT_DIR) if x.isdigit()])

# Cache to store created directories
created_dirs = set()

for year in years:
    print(year)

    files = [x for x in os.listdir(os.path.join(INPUT_DIR, str(year))) if x.endswith('.npy')]

    for file in tqdm(files):
        year_str, month, day, hour = file[:4], file[4:6], file[6:8], file[8:10]

        # Create directory in output_dir for the year and month if it doesn't exist
        output_path = os.path.join(OUTPUT_DIR, year_str, month)
        if output_path not in created_dirs:
            os.makedirs(output_path, exist_ok=True)
            created_dirs.add(output_path)

        # Load and save compressed data
        data = np.load(os.path.join(INPUT_DIR, year_str, file))
        np.savez_compressed(os.path.join(output_path, file.replace('.npy', '.npz')), data=data)