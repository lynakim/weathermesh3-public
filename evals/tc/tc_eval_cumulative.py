import sys
sys.path.append('.')
from utils import *
import functools
from tropycal import tracks
import matplotlib.pyplot as plt
import pickle

# --------------------------
# Customizable parameters 
# --------------------------

FORECAST_START = datetime(2024, 6, 13, 0) 
FORECAST_END = datetime(2024, 8, 31, 0) 

OLD_RUN_NAME = 'bachelor'
# OLD_RUN_TAG = ''
NEW_RUN_NAME = 'operational_stackedconvplus'
NEW_RUN_TAG = 'nt6M'

PHYSICS_MODELS = {
    'OFCL': 'NHC',
}

SAVE_IMAGES = False
REWRITE_DATA = True

# --------------------------
# Constants
# --------------------------

IMAGE_SAVE_PATH = '/huge/users/jack/eval_cumulative/eval_images/'
FORECAST_FILES_PATH = '/huge/users/jack/eval_cumulative/'

TOTAL_MODELS = len(PHYSICS_MODELS) + 2

FORECAST_DAYS = 5
HOUR_SEPARATION = 6
TOTAL_TIME_INDICES = int(FORECAST_DAYS * 24 / HOUR_SEPARATION)

FIG_SIZE = (10, 10)
DPI_VAL = 300

SEARCH_RADIUS = 20 # In pixels for tc models
MESH_SIZE = (720, 1440)

EARTH_RADIUS = 6371 # in km

_GRID_DEGREES = 0.25
_STORM_RADIUS = 1.5 
STORM_RADIUS_IN_CELLS = int(_STORM_RADIUS /_GRID_DEGREES)

# --------------------------
# Decorator functions
# --------------------------
def verbose_evaluate(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        
        assert hasattr(self,'true_lats') and self.true_lats is not None, "True lats not set. Make sure to run evaluate_ground_truth() first."
        assert hasattr(self,'true_lons') and self.true_lons is not None, "True lons not set. Make sure to run evaluate_ground_truth() first."
        assert hasattr(self,'true_vmaxs') and self.true_vmaxs is not None, "True vmaxs not set. Make sure to run evaluate_ground_truth() first."
        assert hasattr(self,'start_time') and self.start_time is not None, "Start time not set. Make sure to run evaluate_ground_truth() first."
        assert hasattr(self,'final_time') and self.final_time is not None, "Final time not set. Make sure to run evaluate_ground_truth() first."
        print(ORANGE(f"Running {func.__name__}..."))
        
        time_start = time.time()
        output = func(self, *args, **kwargs)
        if output == None: 
            print(RED(f"Finished running {func.__name__}. No data found.\n"))
            return
        lats, lons, vmaxs, model_index = output
        time_end = time.time()
        
        # Calculate errors
        track_error, average_track_error = self.calculate_track_error(lats, lons, model_index)
        intensity_error, average_intensity_error = self.calculate_intensity_error(vmaxs, model_index)
        
        # Update class variables
        self.track_errors[:, model_index] += track_error
        self.intensity_errors[:, model_index] += intensity_error
        
        print(f"Average track error: {average_track_error} | Average intensity error: {average_intensity_error}")
        print(GREEN(f"Finished running {func.__name__}. Took {(time_end - time_start) * 1000} ms.\n"))
        
    return wrapper

# --------------------------
# General Helper Functions
# --------------------------
def _find_centroid(data):
    threshold = np.mean(data) * 2
    mask = data > threshold
    points_above = np.where(mask)
    weights = data[mask]
    peak_x = np.average(points_above[0], weights=weights)
    peak_y = np.average(points_above[1], weights=weights)
    return int(round(peak_x)), int(round(peak_y))

def _latlon_to_pixel(lat, lon):
    row = int((90 - lat) / _GRID_DEGREES)
    if lon >= 0: col = int(lon / _GRID_DEGREES)
    else: col = int((360 + lon) / _GRID_DEGREES)
    return row, col

def _pixel_to_latlon(row, col):
    lat = 90 - row * _GRID_DEGREES
    lon = col * _GRID_DEGREES
    if lon > 180:
        lon -= 360
    return lat, lon

# --------------------------
# Main class
# --------------------------
class TCEvalCumulative():
    def __init__(self, new_model_name, old_model_name):
        self.new_model_name = new_model_name
        self.old_model_name = old_model_name
        self.dates = []
        
        # NORMALIZE
        self.normalization = pickle.load(open('/fast/consts/normalization.pickle', "rb"))
        
        self.tracks = tracks.TrackDataset(basin='both', source='hurdat', include_btk=True)
        
        self.track_count = np.zeros((TOTAL_TIME_INDICES, TOTAL_MODELS))
        self.track_errors = np.zeros((TOTAL_TIME_INDICES, TOTAL_MODELS))
        
        self.intensity_count = np.zeros((TOTAL_TIME_INDICES, TOTAL_MODELS))
        self.intensity_errors = np.zeros((TOTAL_TIME_INDICES, TOTAL_MODELS))
        
        self.EXPECTED_HOURS = np.arange(0, FORECAST_DAYS * 24, HOUR_SEPARATION)
        
    # --------------------------
    # Helper functions
    # --------------------------
    # Forecast interval in hours acts as both
    #   The interval between macroscopic forecasts (starting on this time and then moving to this time)
    #   The interval between forecasts started on a date (starting on this time and then predicting the next timestep)
    def generate_dates(self, 
                       forecast_start, 
                       forecast_end, 
                       forecast_interval_in_hours):
        
        self.forecast_start = forecast_start
        self.forecast_end = forecast_end
        working_date = forecast_start
        
        bachelor_files = os.listdir(FORECAST_FILES_PATH + '/bachelor')
        self.bachelor_files = set(bachelor_file[:10] if '.json' in bachelor_file else '' for bachelor_file in bachelor_files)
        new_files = os.listdir(FORECAST_FILES_PATH + '/operational_stackedconvplus')
        self.new_files = set(new_file[:10] if '.json' not in new_file else '' for new_file in new_files)
        
        while working_date <= forecast_end:
            self.verify_date(working_date)
            self.dates.append(working_date)
            working_date += timedelta(hours=forecast_interval_in_hours)
            
    # Verify that the date exists in the forecast models
    def verify_date(self, date):
        str_date = date.strftime('%Y%m%d%H')
        # Verify that the date exists in bachelor 
        if str_date not in self.bachelor_files: raise Exception(f"{str_date} not in bachelor")
        
        # Verify that the date exists in the model
        if str_date not in self.new_files: raise Exception(f"{str_date} not in new model")
           
    # Check if the storm contains all physics forecasts
    def contains_physics_forecasts(self, storm):
        for model in PHYSICS_MODELS:
            if model not in self.forecasts.keys():
                print(f"Physics model {model} not found in storm {storm}.")
                return False
        return True

    # Identify the storm in the bachelor data
    def identify_bachelor_storm(self, bachelor_data):
        start_lat = self.true_lats[0]
        start_lon = self.true_lons[0]
        
        possible_paths = []
        possible_times = []
        possible_vmaxs = []
        storm_keys = []
        
        for storm_key in bachelor_data.keys():
            storm_data = bachelor_data[storm_key]
            storm_keys.append(storm_key)
            
            current_path = []
            current_times = []
            current_vmaxs = []
            
            for storm_dict_elem in storm_data:
                if {'valid_at', 'time', 'latitude', 'longitude', 'max_wind_speed'}.issubset(storm_dict_elem):
                    forecast_time = datetime.fromisoformat(storm_dict_elem["time"]).replace(tzinfo=None)
                    current_path.append((storm_dict_elem['latitude'], storm_dict_elem['longitude']))
                    current_vmaxs.append(storm_dict_elem['max_wind_speed'])
                    current_times.append(forecast_time)
            
            if len(current_path) == 0: continue        
            
            # Sort path based on time      
            sorted_path = []
            sorted_times = []
            sorted_vmaxs = []
            for time, location, vmax in sorted(zip(current_times, current_path, current_vmaxs)):
                sorted_path.append(location)
                sorted_times.append(time)
                sorted_vmaxs.append(vmax)
            possible_paths.append(sorted_path)
            possible_times.append(sorted_times)
            possible_vmaxs.append(sorted_vmaxs)
                        
        if len(possible_paths) == 0: return None
        
        min_distance = float('inf')
        closest_storm_path = None
        closest_storm_times = None
        closest_storm_vmaxs = None
        
        for path, vmaxs, times, storm_key in zip(possible_paths, possible_vmaxs, possible_times, storm_keys):
            path_start_lat = path[0][0]
            path_start_lon = path[0][1]
            distance = np.sqrt((path_start_lat - start_lat)**2 + (path_start_lon - start_lon)**2)
            if distance < min_distance:
                min_distance = distance
                closest_storm_path = path
                closest_storm_times = times
                closest_storm_vmaxs = vmaxs
                
        output_lats = np.array([lat for lat, lon in closest_storm_path])
        output_lons = np.array([lon for lat, lon in closest_storm_path])
        output_times = np.array([int((time - self.start_time).total_seconds() / 3600) for time in closest_storm_times])
        return output_lats, output_lons, np.array(closest_storm_vmaxs), output_times

    # Identifies the storm's next location based on the previous location
    def gather_storm_location(self, tc_maxws_data, previous_x, previous_y):
        data_chunk = tc_maxws_data[previous_x - SEARCH_RADIUS: previous_x + SEARCH_RADIUS, 
                                   previous_y - SEARCH_RADIUS: previous_y + SEARCH_RADIUS]
        x, y = _find_centroid(data_chunk)
        
        actual_x = x + previous_x - SEARCH_RADIUS
        actual_y = y + previous_y - SEARCH_RADIUS
        return actual_x, actual_y

    def gather_vmax(self, data, row, col):
        y, x = np.ogrid[-STORM_RADIUS_IN_CELLS:STORM_RADIUS_IN_CELLS + 1, 
                        -STORM_RADIUS_IN_CELLS:STORM_RADIUS_IN_CELLS + 1]
    
        mask = x*x + y*y <= STORM_RADIUS_IN_CELLS*STORM_RADIUS_IN_CELLS
        
        region = data[row - STORM_RADIUS_IN_CELLS:row + STORM_RADIUS_IN_CELLS + 1,
                      col - STORM_RADIUS_IN_CELLS:col + STORM_RADIUS_IN_CELLS + 1]
        
        masked_data = region[mask]
        return (np.mean(masked_data) * np.sqrt(self.normalization['tc-maxws'][1])) + self.normalization['tc-maxws'][0]

    # Identify the storm in the tc model data
    def identify_tc_storm(self, directory):
        times = []
        lats = []
        lons = []
        vmaxs = []
        
        # Find the pixel location of the start of the storm
        start_lat = self.true_lats[0]
        start_lon = self.true_lons[0]
        current_x, current_y = _latlon_to_pixel(start_lat, start_lon)
        
        for forecast_hour in self.EXPECTED_HOURS:
            tc_maxws = np.load(f'{directory}{forecast_hour}.{NEW_RUN_TAG}.npy', mmap_mode='r')[:,:,-2]
            
            current_x, current_y = self.gather_storm_location(tc_maxws, current_x, current_y)
            storm_lat, storm_lon = _pixel_to_latlon(current_x, current_y)
            vmax = self.gather_vmax(tc_maxws, current_x, current_y)
            
            lats.append(storm_lat)
            lons.append(storm_lon)
            vmaxs.append(vmax)
            times.append(forecast_hour)
        
        return np.array(lats), np.array(lons), np.array(vmaxs), np.array(times)

    # Matches lats, lons, and vmaxs at times to desired hours
    # Times is the time of the forecast data we have that we're attempting to match to gold label desired hours from storm's ground truth data
    def match_forecasts(self, times, lats, lons, vmaxs):
        matched_lats = np.full_like(self.EXPECTED_HOURS, np.nan, dtype=np.float64)
        matched_lons = np.full_like(self.EXPECTED_HOURS, np.nan, dtype=np.float64)
        matched_vmaxs = np.full_like(self.EXPECTED_HOURS, np.nan, dtype=np.float64)
        
        for index, expected_time in enumerate(self.EXPECTED_HOURS):
            if expected_time in times:
                idx = np.where(times == expected_time)[0][0]
                matched_lats[index] = lats[idx]
                matched_lons[index] = lons[idx]
                matched_vmaxs[index] = vmaxs[idx]
        
        return matched_lats, matched_lons, matched_vmaxs

    # Using self.true_lats and self.true_lons, calculate track error over predicted hours        
    def calculate_track_error(self, lats, lons, model_index):
        # Lats and lons can have nans, need to mask them and only calculate error for non-nan values
        mask = np.logical_and(
            np.logical_and(
                ~np.isnan(lats),
                ~np.isnan(lons)
            ),
            np.logical_and(
                ~np.isnan(self.true_lats),
                ~np.isnan(self.true_lons)
            )
        )
        lats = np.where(mask, lats, 0)
        lons = np.where(mask, lons, 0)
        true_lats = np.where(mask, self.true_lats, 0)
        true_lons = np.where(mask, self.true_lons, 0)
        self.track_count[:, model_index] += mask
        
        lat1 = np.radians(true_lats)
        lon1 = np.radians(true_lons)
        lat2 = np.radians(lats)
        lon2 = np.radians(lons)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance_km = EARTH_RADIUS * c # Calculate distance in km
        
        denominator = np.nansum(mask)
        if denominator == 0: denominator = 1e-10
        average_error = np.sum(distance_km) / denominator
        return distance_km, average_error
    
    # Using self.true_vmaxs, calculate intensity error over predicted hours
    def calculate_intensity_error(self, vmaxs, model_index):
        # Need to mask out nan values
        mask = np.logical_and(~np.isnan(vmaxs), ~np.isnan(self.true_vmaxs))
        vmaxs = np.where(mask, vmaxs, 0)
        true_vmaxs = np.where(mask, self.true_vmaxs, 0)
        self.intensity_count[:, model_index] += mask
        error = np.abs(vmaxs - true_vmaxs)
        
        denominator = np.nansum(mask)
        if denominator == 0: denominator = 1e-10
        average_error = np.sum(error) / denominator
        return error, average_error 
        
    # --------------------------
    # End Helper Functions
    # --------------------------    
        
    # Evaluate truth track 
    # Uses tropycal
    def evaluate_ground_truth(self, date, storm):
        print(ORANGE(f"Gathering ground truth data..."))
        storm_times = storm.time
        start_index = np.where(storm.time == date)[0][0]
        self.start_time = storm_times[start_index]
        
        def gather_true_data(storm, storm_times, start_index, end_index):
            self.final_time = storm_times[end_index]
            self.true_lats = storm.lat[start_index: end_index]
            self.true_lons = storm.lon[start_index: end_index]
            self.true_vmaxs = storm.vmax[start_index: end_index]
            self.desired_hours = np.array([int((time - self.start_time).total_seconds() / 3600) for time in storm_times[start_index: end_index]])
            self.true_lats, self.true_lons, self.true_vmaxs = self.match_forecasts(self.desired_hours, self.true_lats, self.true_lons, self.true_vmaxs)
        
        index_diff = min(TOTAL_TIME_INDICES, len(storm_times) - start_index - 1)
        gather_true_data(storm, storm_times, start_index, start_index + index_diff)
        print(GREEN(f"Finished gathering ground truth data.\n"))
    
    # Evaluate physics forecasts
    # Uses tropycal
    @verbose_evaluate
    def evaluate_physics(self, model, model_index):
        
        forecast_times = self.forecasts[model]
        forecast_data = forecast_times[self.start_time.strftime('%Y%m%d%H')]
        
        lats = forecast_data['lat']
        lons = forecast_data['lon']
        vmaxs = forecast_data['vmax']
        times = forecast_data['fhr']
        
        return (*self.match_forecasts(np.array(times), lats, lons, vmaxs), model_index)
    
    # Evaluate bachelor (1st old model)
    # Uses json
    @verbose_evaluate
    def evaluate_bachelor(self, date):
        model_index = -2
        
        # Load in the actual data
        with open(f'{FORECAST_FILES_PATH}bachelor/{date.strftime("%Y%m%d%H")}_tcs.json', 'r') as f:
            bachelor_data = json.load(f)
        
        output = self.identify_bachelor_storm(bachelor_data)
        if output == None: return None
        storm_lats, storm_lons, storm_vmaxs, storm_times = output
        
        return (*self.match_forecasts(storm_times, storm_lats, storm_lons, storm_vmaxs), model_index)
    
    # Evaluate model (new (or old) tc models (once we outperform bachelor))
    # Uses model output (from running deep/realtime/run_TC/run_cumulative_TC.py)
    @verbose_evaluate
    def evaluate_tc_model(self, date):     
        model_index = -1   
        
        # Load in the actual data for the time
        directory = f'{FORECAST_FILES_PATH}{self.new_model_name}/{date.strftime("%Y%m%d%H")}/det/'
        
        storm_lats, storm_lons, storm_vmaxs, storm_times = self.identify_tc_storm(directory)
        
        return (*self.match_forecasts(storm_times, storm_lats, storm_lons, storm_vmaxs), model_index)
    
    def evaluate_date(self, date):
        storms = self.tracks.search_time(date)
        if len(storms) == 0:
            print(f"No storms found for {date}.")
            return
        
        for storm_id in storms['id']:
            storm = self.tracks.get_storm(storm_id)
            self.forecasts = storm.get_operational_forecasts()
            if not self.contains_physics_forecasts(storm): continue
            
            self.evaluate_ground_truth(date, storm)
            for model_index, model in enumerate(PHYSICS_MODELS.keys()):
                self.evaluate_physics(model, model_index)
            
            # --------------------------
            self.evaluate_bachelor(date) 
            self.evaluate_tc_model(date)
            # ---- vs ----
            # self.evaluate_tc_model(self.old_model_name, date)
            # self.evaluate_tc_model(self.new_model_name, date)
            # --------------------------

    def evaluate_models(self):
        assert len(self.dates) > 0, "No dates to evaluate. Make sure to run generate_dates() first."
        
        for date in self.dates:
            print(MAGENTA(f"Starting to evaluate date {date}..."))
            self.evaluate_date(date)
            print(MAGENTA(f"Finished evaluating date {date}.\n"))
        
        print(GREEN(f"🎉🎉🎉 Finished evaluating all dates! 🎉🎉🎉"))
    
    def save_arrays(self):
        # Save data to np array for easy access (so we don't have to rerun this function)
        if REWRITE_DATA:
            print(ORANGE("Saving arrays..."))
            np.save(f'{IMAGE_SAVE_PATH}track_errors.npy', self.track_errors)
            np.save(f'{IMAGE_SAVE_PATH}intensity_errors.npy', self.intensity_errors)
            np.save(f'{IMAGE_SAVE_PATH}track_count.npy', self.track_count)
            np.save(f'{IMAGE_SAVE_PATH}intensity_count.npy', self.intensity_count)
            print(GREEN("Successfully saved arrays!"))
    
    def save_images(self):
        if not SAVE_IMAGES:
            print(RED("Not saving images. Set SAVE_IMAGES to True to save images."))
            return
        
        print(ORANGE("Saving images..."))
        def create_fig_ax(title, ylabel):
            fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI_VAL)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Forecast hour')
            ax.set_xlim([min(self.EXPECTED_HOURS), max(self.EXPECTED_HOURS)])
            ax.grid(visible=True, which='major', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            return fig, ax
        
        self.track_fig, self.track_ax = create_fig_ax(
            f'Average track error (over {self.forecast_start} to {self.forecast_end})', 
            'Average track error (km)'
        )
        
        self.intensity_fig, self.intensity_ax = create_fig_ax(
            f'Average intensity error (over {self.forecast_start} to {self.forecast_end})', 
            'Average intensity error (kts)'
        )
        
        self.count_track_fig, self.count_track_ax = create_fig_ax(
            f'Number of track datapoints (over {self.forecast_start} to {self.forecast_end})', 
            'Number of track datapoints'
        )
        
        self.count_intensity_fig, self.count_intensity_ax = create_fig_ax(
            f'Number of intensity datapoints (over {self.forecast_start} to {self.forecast_end})', 
            'Number of intensity datapoints'
        )
        
        for model_index, model_name in enumerate(list(PHYSICS_MODELS.values()) + [self.old_model_name, self.new_model_name]):
            zero_mask_intensity = ~np.equal(self.intensity_count[:, model_index], 0)
            zero_mask_track = ~np.equal(self.track_count[:, model_index], 0)
            
            plot_intensity_errors = self.intensity_errors[:, model_index][zero_mask_intensity] / self.intensity_count[:, model_index][zero_mask_intensity]
            plot_track_errors = self.track_errors[:,  model_index][zero_mask_track] / self.track_count[:, model_index][zero_mask_track]
            
            self.intensity_ax.plot(self.EXPECTED_HOURS[zero_mask_intensity], plot_intensity_errors, label=model_name, marker='o')
            self.track_ax.plot(self.EXPECTED_HOURS[zero_mask_track], plot_track_errors, label=model_name, marker='o')
            
            self.count_intensity_ax.plot(self.EXPECTED_HOURS, self.intensity_count[:, model_index], label=model_name, marker='o')
            self.count_track_ax.plot(self.EXPECTED_HOURS, self.track_count[:, model_index], label=model_name, marker='o')
            
        # Create legends
        self.intensity_ax.legend()
        self.track_ax.legend()
        self.count_intensity_ax.legend()
        self.count_track_ax.legend()    
            
        # Save the images
        os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
        self.intensity_fig.savefig(f'{IMAGE_SAVE_PATH}intensity_error.png')
        self.track_fig.savefig(f'{IMAGE_SAVE_PATH}track_error.png')
        self.count_intensity_fig.savefig(f'{IMAGE_SAVE_PATH}count_intensity.png')
        self.count_track_fig.savefig(f'{IMAGE_SAVE_PATH}count_track.png')
        print(GREEN("Successfully saved images!"))

if __name__ == '__main__':
    
    new_run_name = NEW_RUN_NAME
    old_run_name = OLD_RUN_NAME
    
    tc_eval = TCEvalCumulative(new_run_name, old_run_name)
    
    tc_eval.generate_dates(
        forecast_start=FORECAST_START,
        forecast_end=FORECAST_END,
        forecast_interval_in_hours=HOUR_SEPARATION
    )
    try:
        tc_eval.evaluate_models()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Saving current results...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Attempting to save results...")
        try:
            tc_eval.save_arrays()
            tc_eval.save_images()
        except Exception as e:
            print(f"An error occurred while saving results: {e}")