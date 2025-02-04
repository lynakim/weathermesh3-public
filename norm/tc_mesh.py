import sys
sys.path.append('..')
import os
from utils import *
from utils_lite import *
from tropycal import tracks

# Type/Naming Constants 
TC_CONSTANTS = {
    'wind' : 'tc-maxws',
    'pressure' : 'tc-minp'
}

class TCMeshGatherer:
    def __init__(self, verbose=True, rewriting=False, write_nans=True, should_save=False, verbose_debugging=False):
        # Save paths
        self.save_path = '/fast/proc/era5/extra'
        self.tc_location_save_path = '/fast/proc/cyclones/locations'
        self.tc_intensities_save_path = '/fast/proc/cyclones/intensities'
        
        # Parameter information 
        self.verbose = verbose
        self.verbose_debugging = verbose_debugging
        self.write_nans = write_nans
        self.rewriting = rewriting
        self.should_save = should_save
        
        # Constants
        self.mesh_size = (720, 1440)
        self.grid_degrees = 0.25 # Size in degrees of each grid cell
        self.storm_radius = 1.5 # ~ 1.5 degree radius / ~ 100 nautical miles assumed for now (generally an overestimation)
        self.default_threshold = 100 # Threshold of number of default values that need to be placed in NaN files, about the size of one cyclone (113)
        self.default_wind = 0 # Default wind speed value
        self.default_pressure = 1009.5 # Default pressure value in hPa (~1 atm, empirically found via ERA5/f000 mslp)
        
        # TODO RADIUS INFORMATION / Assume radius constant even through map distortion
        self.storm_radius_in_cells = int(self.storm_radius / self.grid_degrees)
        
        # Gather tropycal tracking data (+ max wind speed and min pressure data)
        print("Reading in IBTrACS data from WMO with catarina flag.")
        self.track = tracks.TrackDataset(basin='all', 
                                         source='ibtracs',
                                         ibtracs_mode='wmo',
                                         catarina=True)
        
        # Make the TC directories
        os.makedirs(os.path.join(self.save_path, TC_CONSTANTS['wind']), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, TC_CONSTANTS['pressure']), exist_ok=True)

        # Tracking
        self.missed = 0
        if self.verbose_debugging: self.randomx = []
        if self.verbose_debugging: self.randomy = []
    
    def gather_path_date_suffix(self, date):
        year_month = f"%04d%02d" % (date.year, date.month)
        unix_date = f"%d.npz" % to_unix(date)
        return year_month, unix_date
        
    def gather_path(self, type, date):
        year_month, unix_date = self.gather_path_date_suffix(date)
        year_month_directory = f"{self.save_path}/%s/{year_month}" % type
        file_path = f"{year_month_directory}/{unix_date}"
        
        if self.verbose_debugging: print(year_month_directory)
            
        return year_month_directory, file_path
    
    def should_continue(self, type, date):
        _, file_path = self.gather_path(type, date)
        return os.path.exists(file_path) and not self.rewriting
    
    # Fills mesh with default values at random NaN locations
    def fill_default_values(self, mesh, default_value, existing_values):
        threshold = self.default_threshold
        if threshold < len(existing_values): threshold = len(existing_values)
        
        for _ in range(threshold):
            i = np.random.randint(0, self.mesh_size[0])
            j = np.random.randint(0, self.mesh_size[1])
            while (i, j) in existing_values:
                print("Found duplicate value. Retrying...")
                i = np.random.randint(0, self.mesh_size[0])
                j = np.random.randint(0, self.mesh_size[1])
                
            if self.verbose_debugging: self.randomx.append(i)
            if self.verbose_debugging: self.randomy.append(j)
            
            mesh[i, j] = default_value
            existing_values.add((i, j))
    
    def save_extra(self, date, extra, path):
        year_month, unix_date = self.gather_path_date_suffix(date)
        year_month_directory = f"{path}/{year_month}"
        file_path = f"{year_month_directory}/{unix_date}"
        
        os.makedirs(year_month_directory, exist_ok=True)
        
        print("Saving extra data to %s..." % file_path)
        np.savez(file_path, x=np.array(extra).astype(np.float32))
    
    # Calculates the pressure average for a given date
    def calculate_pressure_average(self, date):
        return self.default_pressure
    
    # Saves a NaN array for a given date (with dummy values used)
    def save_nan(self, type, date):
        mesh = np.full(self.mesh_size, np.nan)
        nonNaNdefault = self.default_wind if type == TC_CONSTANTS['wind'] else self.calculate_pressure_average(date)
        # Logic to fill the mesh with default values at random locations
        self.fill_default_values(mesh, nonNaNdefault, set())
        assert np.sum(~np.isnan(mesh)) == self.default_threshold, f"Not enough default values for NaN array"
        self.save_mesh(type, date, mesh)
    
    def gather_mesh(self, date):
        # Early stopping in case the file already exists
        if self.should_continue(TC_CONSTANTS['wind'], date) and self.should_continue(TC_CONSTANTS['pressure'], date):
            print(f"Skipping {date}. Already exists.")
            return
        
        # Gather wind speed or pressure information from tropycal 
        storms = self.track.search_time(date.replace(tzinfo=None))
        
        mesh_wind = np.full(self.mesh_size, np.nan)
        mesh_pressure = np.full(self.mesh_size, np.nan)
        
        if storms.empty:
            if self.write_nans:
                print(f"No storms on {date}. {'Writing NaNs.' if self.should_save else ''}")
                self.save_nan(TC_CONSTANTS['wind'], date)
                self.save_nan(TC_CONSTANTS['pressure'], date)
                self.save_extra(date, [], self.tc_location_save_path)
                self.save_extra(date, [], self.tc_intensities_save_path)
                return
            else:
                self.missed += 1
                print(f"No storms on {date}. Skipping.")
                return
        
        # Iterate over all storms
        storm_locations = set()
        storm_intensities = []
        storm_row_col = []
        for _, storm in storms.iterrows():
            if self.verbose: print(storm['name'])
            
            storm_lat = storm['lat']
            storm_lon = storm['lon']
            storm_vmax = storm['vmax']
            storm_mslp = storm['mslp']
            
            # Make sure both values are present
            if np.isnan(storm_mslp) or np.isnan(storm_vmax): 
                print(f"Missing {'mslp' if np.isnan(storm_mslp) else 'vmax'} data for {storm['name']} {storm['id']} on {date}. Skipping.")
                continue
            
            # Parse storm location on mesh
            row = int((90 - storm_lat) / self.grid_degrees)
            if storm_lon >= 0: col = int(storm_lon / self.grid_degrees)
            else: col = int((360 + storm_lon) / self.grid_degrees)
            
            storm_row_col.append((row, col))
            storm_intensities.append((storm_vmax, storm_mslp))
            
            for i in range(row - self.storm_radius_in_cells, row + self.storm_radius_in_cells + 1):
                for j in range(col - self.storm_radius_in_cells, col + self.storm_radius_in_cells + 1):
                    # Boundary checking (and boundary wrapping)
                    if i < 0: i = self.mesh_size[0] - i
                    elif i >= self.mesh_size[0]: i = i - self.mesh_size[0]
                    elif j < 0: j = self.mesh_size[1] - j
                    elif j >= self.mesh_size[1]: j = j - self.mesh_size[1]
                    
                    # Circle boundary checking
                    if (i - row) ** 2 + (j - col) ** 2 > self.storm_radius_in_cells ** 2: continue
                    
                    mesh_wind[i, j] = storm_vmax
                    mesh_pressure[i, j] = storm_mslp
                    storm_locations.add((i, j))
            
        # Add random values to NaNs
        self.fill_default_values(mesh_wind, self.default_wind, storm_locations.copy())
        self.fill_default_values(mesh_pressure, self.calculate_pressure_average(date), storm_locations.copy())
        
        assert len(storm_locations) == 0 or np.sum(~np.isnan(mesh_wind)) == 2 * len(storm_locations), f"Not enough default wind values"
        assert len(storm_locations) == 0 or np.sum(~np.isnan(mesh_pressure)) == 2 * len(storm_locations), f"Not enough default pressure values"
        assert np.sum(~np.isnan(mesh_wind)) == np.sum(~np.isnan(mesh_pressure)), f"Wind and pressure values don't match"
        
        if len(storm_row_col) > 0:
            self.save_mesh(TC_CONSTANTS['wind'], date, mesh_wind)
            self.save_mesh(TC_CONSTANTS['pressure'], date, mesh_pressure)
            self.save_extra(date, storm_row_col, self.tc_location_save_path)
            self.save_extra(date, storm_intensities, self.tc_intensities_save_path)
            return 
        else:
            if self.write_nans:
                print(f"No valid storms on {date}. Caused by missing data. {'Writing NaNs.' if self.should_save else ''}")
                self.save_nan(TC_CONSTANTS['wind'], date)
                self.save_nan(TC_CONSTANTS['pressure'], date)
                self.save_extra(date, [], self.tc_location_save_path)
                self.save_extra(date, [], self.tc_intensities_save_path)
                return
            else:
                self.missed += 1
                print(f"No valid storms on {date}. Caused by missing data. Skipping.")
                return 
        
    def save_mesh(self, type, date, data):
        # Early stopping in case we don't want to save
        if not self.should_save: return
        
        # Create year/month directory
        year_month_directory, file_path = self.gather_path(type, date)
        
        os.makedirs(year_month_directory, exist_ok=True)
            
        if self.verbose_debugging: print("File path: ", file_path)
        assert os.path.join('/', *file_path.split('/')[:-3]) == self.save_path, f"Save path is not in the correct directory"
        assert file_path.split('/')[-3] == type, f"Type is not in the correct directory"
        
        # Save the data
        print_vars = file_path.split('/')[-2:]
        print(f"Saving {type} data to {print_vars[0]}/{print_vars[1]}...")
        np.savez(file_path, x=data)
        
if __name__ == '__main__':
    verbose = True              # If we should print out status statements
    verbose_debugging = False   # Debugging for random x and y values and other print statements
    rewriting = True            # If we should rewrite files
    use_small_dates = False     # If we should only run a small subset of dates
    write_nans = True           # If we should write NaN files
    should_save = False         # If we should save files
    
    mesh_gatherer = TCMeshGatherer(verbose, rewriting, write_nans, should_save, verbose_debugging)
    
    # Good test times
    # D(2020, 9, 15) -> five storms at once
    # D(2014, 5, 10) -> no storms
    # D(1851, 6, 25) -> no pressure data
    dates = get_dates([(D(1971, 1, 1), D(2024, 1, 1), timedelta(hours=3))])
    if use_small_dates: dates = get_dates([(D(2001, 9, 1), D(2001, 10, 10), timedelta(hours=3))])
    
    for date in dates:
        if verbose: print(date.year)
        
        mesh_gatherer.gather_mesh(date)
    
        if verbose: print(' ')
        
    if verbose_debugging:
        plt.hist(mesh_gatherer.randomx, bins=mesh_gatherer.mesh_size[0])
        plt.savefig('/huge/users/jack/x.png')
        
        plt.hist(mesh_gatherer.randomy, bins=mesh_gatherer.mesh_size[1])
        plt.savefig('/huge/users/jack/y.png')
    
    print("Number of missed storm dates: ", mesh_gatherer.missed)