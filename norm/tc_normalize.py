import sys
sys.path.append('..')
import os
from utils import *
import numpy as np
import pickle

TC_CONSTANTS = {
    'wind' : 'tc-maxws',
    'pressure' : 'tc-minp'
}

# Load pickle
class TC_Normalization():
    def __init__(self, var_name):
        self.save_path = '/fast/proc/era5/extra'
        self.pickle_path = f"{CONSTS_PATH}/normalization.pickle"
        
        with open(self.pickle_path, "rb") as f:
            self.pickle = pickle.load(f)
        
        # Make sure normalization is not empty
        assert self.pickle, "Pickle is empty."

        # Debugging dates to use (9 days)  
        # self.dates = get_dates([(D(2019, 1, 1), D(2019, 6, 9), timedelta(hours=3))]) 
        # Two years of data for normalization (2019-2020)
        self.dates = get_dates([(D(2019, 1, 1), D(2021, 1, 1), timedelta(hours=3))])
        
        self.average = None
        self.variance = None
        self.var_name = var_name
    
    def load_pickle(self):
        self.average, self.variance = self.pickle[self.var_name]
    
    def save_pickle(self):
        self.pickle[self.var_name] = (np.array([self.average]), np.array([self.variance]))
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.pickle, f)
    
    def gather_data(self, date):
        file_path = f"{self.save_path}/%s/%04d%02d/%d.npz" % (self.var_name, date.year, date.month, to_unix(date))
        assert os.path.exists(file_path), f"File does not exist: {file_path}"
        return file_path
    
    def calculate_norm(self):
        total_sum = 0
        total_count = 0
        last_percent = -1
        
        # Find the average
        print("Beginning average finding for ", self.var_name)
        for i, date in enumerate(self.dates):
            data = np.load(self.gather_data(date))['x']
            
            count = np.sum(~np.isnan(data))
            if count == 0:
                # Found a file with no relevant data
                raise Exception("Found a file with no relevant data.")
            
            sum = np.nansum(data)
            total_sum += sum
            total_count += count
            
            # Print every 20% of the way
            percent = int((i / len(self.dates)) * 100)
            if percent % 20 == 0 and last_percent != percent:
                print(f"{percent}% done with finding average for {self.var_name}")
                last_percent = percent
        
        assert total_count != 0, "Total count is 0 for average."
        self.average = total_sum / total_count
        print("Average found for ", self.var_name)
        
        total_sum = 0
        total_count = 0
        last_percent = -1
        
        # Find the variance 
        print("Beginning variance finding for ", self.var_name)
        for i, date in enumerate(self.dates):
            data = np.load(self.gather_data(date))['x']
            
            count = np.sum(~np.isnan(data))
            if count == 0:
                # Found a file with no relevant data
                raise Exception("Found a file with no relevant data.")
            
            modified_data = np.where(~np.isnan(data), (data - self.average) ** 2, np.nan)
            sum = np.nansum(modified_data)
            total_sum += sum
            total_count += count
            
            # Print every 20% of the way
            percent = int((i / len(self.dates)) * 100)
            if percent % 20 == 0 and last_percent != percent:
                print(f"{percent}% done with finding variance for {self.var_name}")
                last_percent = percent
            
        assert total_count != 0, "Total count is 0 for variance."
        # Using a sample so we divide by n - 1
        self.variance = total_sum / total_count - 1
        print(self.average)
        print(self.variance)

    def update_mesh(self):
        assert self.average != None, "Average is not set."
        assert self.variance != None, "Variance is not set."
        assert self.variance != 0, "Variance is 0."
        
        print("Updating mesh for ", self.var_name)
        path = f"{self.save_path}/{self.var_name}"
        
        files = set()
        _, subfolders, _ = next(os.walk(path))
        for subfolder in subfolders:
            _, _, filenames = next(os.walk(f"{path}/{subfolder}"))
            for filename in filenames:
                # Handle files
                file_path = f"{path}/{subfolder}/{filename}"
                if file_path in files: raise Exception("Duplicate file found.")
                files.add(file_path)
                
                data = np.load(file_path)['x']
                
                # Normalize the data
                data = np.where(~np.isnan(data), (data - self.average) / np.sqrt(self.variance), np.nan)
                data_fp16 = data.astype(np.float16)
                
                # Save the normalized data
                np.savez(file_path, x=data_fp16)
        
                
def normalize_intensities(use_pickle=True):
    
    def gather_data(path, date):
        file_path = f"{path}/%04d%02d/%d.npz" % (date.year, date.month, to_unix(date))
        assert os.path.exists(file_path), f"File does not exist: {file_path}"
        return file_path
    
    path = '/fast/proc/cyclones/intensities'
    pickle_path = '/fast/consts/tc-intensities.pickle'
    dates = get_dates([(D(2010, 1, 1), D(2020, 1, 1), timedelta(hours=3))])
    
    if not use_pickle:
        # Calculate average and variance for wind speed / pressure, place them in 
        print("Starting averages for tc intensities... ")
        count = 0
        sum = np.array([0., 0.])
        for date in dates:
            file_path = gather_data(path, date)
            data = np.load(file_path)['x']
            
            for datum in data:
                count += 1
                sum += datum
                
        averages = sum / count 
        
        print("Starting variances for tc intensities... ")
        count = 0
        sum = np.array([0., 0.])
        for date in dates:
            file_path = gather_data(path, date)
            data = np.load(file_path)['x']
            
            for datum in data:
                count += 1
                sum += (datum - averages) ** 2
        
        variances = sum / count - 1
        
        # Place variances in pickle
        intensity_pickle = {}
        intensity_pickle['wind'] = (np.array([averages[0]]), np.array([variances[0]]))
        intensity_pickle['pressure'] = (np.array([averages[1]]), np.array([variances[1]]))
        
        print("Averages: ", averages)
        print("Variances: ", variances)
        
        with open(pickle_path, "wb") as f:
            pickle.dump(intensity_pickle, f)
    
    # Load pickle
    with open(pickle_path, "rb") as f:
        intensity_pickle = pickle.load(f)
    
    averages = np.array([intensity_pickle['wind'][0], intensity_pickle['pressure'][0]]).squeeze()
    variances = np.array([intensity_pickle['wind'][1], intensity_pickle['pressure'][1]]).squeeze()
    assert variances.shape == (2,), f"Variances shape is not (2,1): {variances.shape}"
    assert averages.shape == (2,), f"Averages shape is not (2,1): {averages.shape}"
    
    print("Starting normalization for tc intensities... ")
    files = set()
    _, subfolders, _ = next(os.walk(path))
    for subfolder in subfolders:
        print("Starting ", subfolder)
        not_saved = 0
        _, _, filenames = next(os.walk(f"{path}/{subfolder}"))
        for filename in filenames:
            # Handle files
            file_path = f"{path}/{subfolder}/{filename}"
            if file_path in files: raise Exception("Duplicate file found.")
            files.add(file_path)
            
            should_save = True
            with np.load(file_path) as data:
                data = data['x']
                save_data = []
                for datum in data:
                    # Quick check to see if the data has already been normalized
                    if datum[1] < 600:
                        should_save = False
                    datum = (datum - averages) / np.sqrt(variances)
                    save_data.append(datum)
                save_data = np.array(save_data)
                save_data_fp16 = save_data.astype(np.float16)
            
            # Prevents race condition (Errno 116)
            time.sleep(0.01)
            
            # Save the normalized data
            if should_save:
                np.savez(file_path, x=save_data_fp16)
            else:
                not_saved += 1
        print(f"{subfolder} done.")
        print("Not saved: ", not_saved)
    
if __name__ == '__main__':
    #normalize_intensities()
    
    tc_normalization_wind = TC_Normalization(TC_CONSTANTS['wind'])
    tc_normalization_wind.calculate_norm()
    tc_normalization_wind.save_pickle()
    tc_normalization_wind.update_mesh()
    
    tc_normalization_pressure = TC_Normalization(TC_CONSTANTS['pressure'])
    tc_normalization_pressure.calculate_norm()
    tc_normalization_pressure.save_pickle()
    tc_normalization_pressure.update_mesh()
    
    