import sys
sys.path.append('.')
from utils import *
import shutil 
from tropycal import tracks
from matplotlib.colors import ListedColormap, Normalize
from PIL import Image

# CONSTANTS 
MEAN = {
    'wind': 25.38431485,
    'pressure': 999.64465439
}
VARIANCE = {
    'wind': 1052.03883192,
    'pressure': 284.5727113
}
# Models below
MODELS = {
    #'TCfullforce': '.PINA.npy',
    #'TCvarweight': '.Yfm2.npy',
}
REGIONAL_MODELS = {
    'TCregional': '.0O0B.npy',
    'TCregionalio': '.tbDQ.npy',
}
BACHELOR_MODEL = {
    'bachelor': '.bsxb.npy'
}
# https://web.uwm.edu/hurricane-models/models/models.html
PHYSICS_MODELS = {
    'EMXI': 'IFS', 
    'AVNO': 'GFS',
    'OFCL': 'NHC',
    'JGSI': 'JMA',
    'COTC': 'US Navy COAMPS-TC'
}
ACTUAL_MODEL = 'Ground Truth'
# CONSTANTS

class TC_Eval:
    def __init__(self, storm_name, forecast, forecast_start, forecast_end, start_time, end_time, bounding_box, start_location):
        self.forecast = forecast
        self.forecast_start = forecast_start
        self.forecast_end = forecast_end
        self.start_time = start_time
        self.end_time = end_time
        self.dates = get_dates((self.start_time, self.end_time, timedelta(hours=6)))
        
        self.storm_name = storm_name 
        self.bounding_box = bounding_box
        self.start_location = start_location
        self.search_threshold = 20 # Number of pixels manhattan dist we'll search around the previous location
        self.conversion_factor = 1.94384 # Converts wind speed (m/s) to knots
        self.eval_path = '/huge/users/jack/evals/'
        self.save_path = f'{self.eval_path}/eval_plots/{self.storm_name}/{self.forecast}'
        
        # Regional Intensity normalization
        with open('/fast/consts/tc-intensities.pickle', "rb") as f:
            self.intensity_pickle = pickle.load(f)
        
        # CAUTION Taken from TC MESH
        self.mesh_size = (720, 1440)
        self.grid_degrees = 0.25
        self.storm_radius = 1.5 
        self.storm_radius_in_cells = int(self.storm_radius / self.grid_degrees)
        
        print("Reading in HURDAT data (required for NHC forecasts).")
        self.track = tracks.TrackDataset(basin='both', source='hurdat', include_btk=True)
        '''
        print("Reading in IBTrACS data from WMO with catarina flag.")
        self.track = tracks.TrackDataset(basin='all', 
                                         source='ibtracs',
                                         ibtracs_mode='wmo',
                                         catarina=True)
        '''
        self.storm = self.track.get_storm((self.storm_name.lower(), int(self.forecast[:4])))
        
        # Background earth data
        self.earth_background = np.load('/fast/proc/era5/extra/034_sstk/197601/189723600.npz')['x'].squeeze()
        self.earth_background = self.earth_background[bounding_box['min_y']:bounding_box['max_y'], bounding_box['min_x']:bounding_box['max_x']]
        self.earth_background = np.where(~np.isnan(self.earth_background), 1, 0)
        
        fig_size = (10, 10)
        colors = ['skyblue', 'mediumpurple', 'palegreen', 'limegreen', 'yellow', 'orange', 'red']
        self.cmap = ListedColormap(colors)
        self.norm = Normalize(vmin=-1, vmax=5)
        self.track_fig, self.track_ax = plt.subplots(figsize=fig_size, dpi=300)
        
        self.intensity_track_plts = {ACTUAL_MODEL : plt.subplots(figsize=fig_size, dpi=300)}
        self.intensity_track_plts[ACTUAL_MODEL][1].set_title(f'{ACTUAL_MODEL} Intensity Track')
        for model in (MODELS | BACHELOR_MODEL | REGIONAL_MODELS | PHYSICS_MODELS).keys():
            self.intensity_track_plts[model] = plt.subplots(figsize=fig_size, dpi=300)
            if model in PHYSICS_MODELS.keys():
                self.intensity_track_plts[model][1].set_title(f'{self.storm_name} \n {PHYSICS_MODELS[model]} intensity on path \n Forecasted at {self.forecast_start}')
            else:
                self.intensity_track_plts[model][1].set_title(f'{self.storm_name} \n {model} intensity on path \n Forecasted at {self.forecast_start}')
        
        for _, ax in self.intensity_track_plts.values():
            ax.imshow(self.earth_background, zorder=0, cmap=ListedColormap(['gray', 'lightblue']))
        self.track_ax.imshow(self.earth_background, zorder=0, cmap=ListedColormap(['gray', 'lightblue']))
        self.track_ax.set_title(f'{self.storm_name} Tracking \n Forecasted at {self.forecast_start}')
        
        # Intensity RMSE over time for different models (using true location)
        fig_size = (12, 8)
        self.intensity_at_location_fig, self.intensity_at_location_ax = plt.subplots(figsize=fig_size, dpi=300)
        self.windspeed_at_path_fig, self.windspeed_at_path_ax = plt.subplots(figsize=fig_size, dpi=300)
        
        number_of_values = 12
        end_hour = int((self.end_time - self.start_time).total_seconds() // 3600)
        x_values = [int(i) for i in np.linspace(0, end_hour, number_of_values + 1)]
        
        #x_ticks = [f'+{(date - self.start_time).total_seconds() // 3600}z' for date in self.dates]
        #x_values = [(date - self.start_time).total_seconds() // 3600 for date in self.dates]
        x_label = "Forecast Hour" 
        y_label = "Windspeed (knots)"
        self.intensity_at_location_ax.set_xlabel(x_label) 
        self.intensity_at_location_ax.set_ylabel(y_label)
        self.intensity_at_location_ax.set_xticks(x_values)
        self.intensity_at_location_ax.set_title(f'{self.storm_name} \n Predicted Windspeeds (using locations at Ground Truth) \n Forecasted at {self.forecast_start}')
        self.intensity_at_location_ax.grid(visible=True, which='major', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        self.intensity_at_location_ax.set_axisbelow(True)
        self.windspeed_at_path_ax.set_xlabel(x_label)
        self.windspeed_at_path_ax.set_ylabel(y_label)
        self.windspeed_at_path_ax.set_xticks(x_values)
        self.windspeed_at_path_ax.set_title(f'{self.storm_name} \n Predicted Windspeeds \n Forecasted at {self.forecast_start}')
        self.windspeed_at_path_ax.grid(visible=True, which='major', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        self.windspeed_at_path_ax.set_axisbelow(True)

        self.chunk_size = 40
        self.save_chunks = True
        self.chunk_indices = {model_name: 0 for model_name in (MODELS | BACHELOR_MODEL).keys()}
        self.chunk_file_paths = {model_name: [] for model_name in (MODELS | BACHELOR_MODEL).keys()}
        
    # HELPER Functions
    def _plot_track(self, ax, track_x, track_y, label, intensity=None):
        ax.plot(track_x, 
                track_y, 
                linestyle='-',
                linewidth=2.5,
                zorder=1,
                alpha=0.5, 
                label=label)
        
        scatter_kwargs = {}
        if intensity is not None:
            scatter_kwargs['c'] = intensity
            scatter_kwargs['cmap'] = self.cmap
            scatter_kwargs['norm'] = self.norm
        
        sc = ax.scatter(track_x,
                    track_y,
                    edgecolors='black',
                    linewidths=0.5,
                    zorder=2,
                    **scatter_kwargs) 
        
        if intensity is not None:
            cbar = ax.figure.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, spacing='proportional')
            cbar.set_label('Saffir-Simpson Intensity Scale', fontsize=10)
            cbar.set_ticks(range(-1, 6))
            cbar.set_ticklabels(['TD', 'TS', 'C1', 'C2', 'C3', 'C4', 'C5'])
            
            bbox = ax.get_position()
            cbar.ax.set_position([bbox.x1 + 0.02, bbox.y0 + 0.1, 0.02, bbox.height - 0.2]) 
        
        ax.grid(visible=True, which='major', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        ax.legend()
        
    # Plots GIFs of chunks
    def _plot_chunk(self, model, data, date, location, previous_location):
        x, y = location
        chunk_fig, chunk_ax = plt.subplots()
        
        chunk_ax.imshow(data[previous_location[0] - self.chunk_size:previous_location[0] + self.chunk_size,
                             previous_location[1] - self.chunk_size:previous_location[1] + self.chunk_size]) 
        
        chunk_ax.plot(x + (self.chunk_size - self.search_threshold), y + (self.chunk_size - self.search_threshold), 'ro')
        rect = plt.Rectangle(
            (self.chunk_size - self.search_threshold, self.chunk_size - self.search_threshold),  # Bottom-left corner
            2 * self.search_threshold,  # Width
            2 * self.search_threshold,  # Height
            linewidth=1, edgecolor='red', facecolor='none'
        )
        chunk_ax.add_patch(rect)
        chunk_ax.set_title(f"{self.storm_name} \n Model: {model} \n Date: {date}", fontsize=12)
        chunk_ax.axis('off')
        
        file_path = f'{self.save_path}/temp_gif/{model}_chunk_{self.chunk_indices[model]}.png'
        chunk_fig.savefig(file_path)
        self.chunk_file_paths[model].append(file_path)
        self.chunk_indices[model] += 1
        plt.close(chunk_fig)
        
    def _get_difference_in_hours(self, first, second):
        difference = first - second
        return np.abs((difference.days * 24) + (difference.seconds // 3600))
            
    # Find the closest location to the previous location
    def _gather_nearest_location(self, previous_location, possible_locations):
        if len(possible_locations) == 0:
            return previous_location
        
        for location in possible_locations:
            if np.linalg.norm(np.array(location) - np.array(previous_location)) < self.search_threshold:
                return [int(elem) for elem in location]
        return [int(elem) for elem in possible_locations[0]]
        
    # CAUTION Taken from TC Mesh
    def _convert_lat_lon_to_xy(self, lat, lon):
        row = int((90 - lat) / self.grid_degrees)
        if lon >= 0: col = int(lon / self.grid_degrees)
        else: col = int((360 + lon) / self.grid_degrees)
        return row, col
    
    def remove_temp_files(self):
        print("Removing temp gif folder...")
        shutil.rmtree(f'{self.save_path}/temp_gif')
    
    # Saffir-Simpson scale
    def convert_wind_to_SSintensity(self, wind):
        if wind < 35: return -1
        if wind < 64: return 0
        if wind < 83: return 1
        if wind < 96: return 2
        if wind < 113: return 3
        if wind < 137: return 4
        return 5
    
    # Gathers true windspeed from tc-maxws data using location
    def _gather_windspeed_true(self, location, date):
        intensity_data = np.load(f'/fast/proc/era5/extra/tc-maxws/{date.year}{date.month:02d}/{to_unix(date)}.npz')['x']
        wind = intensity_data[int(location[0]), int(location[1])]
        assert np.isnan(wind) == False, "Wind is nan"
        unnormalized_wind = wind * np.sqrt(VARIANCE['wind']) + MEAN['wind']
        return unnormalized_wind
    
    # Gathers intensity information using data and location in that data
    # Assumes data is average over all mask locations
    def gather_windspeed_model(self, data, location):
        row = location[0]
        col = location[1]
        # CAUTION Iteration functionality is taken from TC MESH (duplicated)
        # CAUTION Assuming output data is unnormalized prediction
        sum = 0
        count = 0
        for i in range(row - self.storm_radius_in_cells, row + self.storm_radius_in_cells + 1):
            for j in range(col - self.storm_radius_in_cells, col + self.storm_radius_in_cells + 1):
                # Boundary checking (and boundary wrapping)
                if i < 0: i = self.mesh_size[0] - i
                elif i >= self.mesh_size[0]: i = i - self.mesh_size[0]
                elif j < 0: j = self.mesh_size[1] - j
                elif j >= self.mesh_size[1]: j = j - self.mesh_size[1]
                
                # Circle boundary checking
                if (i - row) ** 2 + (j - col) ** 2 > self.storm_radius_in_cells ** 2: continue
                
                sum += data[i,j]
                count += 1
                
        assert count != 0, "Divide by zero"
        assert count == 113, "Count is not 113"
        return sum / count
    
    # Parse location from data and previous location
    # Use maximum for the location
    def parse_location_model(self, data, previous_location, model, date):
        
        # HELPER function that takes in data
        def _find_middle(data):
            # Normalize data to be between 0 and 1
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            data_sum = np.sum(data)
            sum = 0
            for i, datapoint in enumerate(data):
                sum += datapoint
                if sum > data_sum / 2:
                    return i
        
        def _find_max(data):
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            return np.argmax(data, axis=None)
        
        def _find_centroid(data):
            coords = np.arange(len(data))
            weights = data / np.sum(data)
            return int(np.sum(coords * weights))
        
        data_chunk = data[previous_location[0] - self.search_threshold:previous_location[0] + self.search_threshold,
                          previous_location[1] - self.search_threshold:previous_location[1] + self.search_threshold]
        cols_collapsed = np.sum(data_chunk, axis=1)
        rows_collapsed = np.sum(data_chunk, axis=0)
        
        x, y = _find_centroid(cols_collapsed), _find_centroid(rows_collapsed)
        
        if self.save_chunks:
            self._plot_chunk(model, data, date, (x, y), previous_location)
        
        return x + previous_location[0] - self.search_threshold, y + previous_location[1] - self.search_threshold
        
    # Expects data in the form of bachelor (one low and one high location)
    def parse_location_bachelor(self, data, previous_location, model, date, prev_loc=None):
        data_chunk = data[previous_location[0] - self.search_threshold:previous_location[0] + self.search_threshold,
                          previous_location[1] - self.search_threshold:previous_location[1] + self.search_threshold]
        min = np.unravel_index(np.argmin(data_chunk), data_chunk.shape)
        max = np.unravel_index(np.argmax(data_chunk), data_chunk.shape)
        x_avg = (min[0] + max[0])/2
        y_avg = (min[1] + max[1])/2
        
        if self.save_chunks and prev_loc:
            x = (x_avg + prev_loc[0])/2
            y = (y_avg + prev_loc[1])/2
            self._plot_chunk(model, data, date, (x, y), previous_location)
            
        return (x_avg + previous_location[0] - self.search_threshold, y_avg + previous_location[1] - self.search_threshold), (np.abs(data_chunk[min]) + data_chunk[max])/2, (x_avg, y_avg)
 
    def parse_location_regional_model(self, previous_location, data):
        # Find closes datum to previous location where location in datum is the 2nd and 3rd index
        datum_distances = []
        for datum in data:
            datum_distances.append(np.linalg.norm(np.array(datum[-2:]) - np.array(previous_location)))
        index = np.argmin(datum_distances)
        maxws = data[index][0]
        normalized_maxws = (maxws * np.sqrt(self.intensity_pickle['wind'][1])) + self.intensity_pickle['wind'][0]
        return normalized_maxws, data[index][-2:]
 
    # Gathers wind for a specific model type
    def gather_wind_data(self, model, file_name, date):
        difference_in_hours = self._get_difference_in_hours(self.forecast_start, date)
        file_path = f'{self.eval_path}/{model}/{self.forecast}/det/{difference_in_hours}{"_intensity" if model in REGIONAL_MODELS.keys() else ""}{file_name}'
        assert os.path.exists(file_path), f"File does not exist: {file_path}"
        if model in BACHELOR_MODEL.keys():
            wind_data_u = np.load(file_path)[:,:,74]
            wind_data_v = np.load(file_path)[:,:,99]
            return wind_data_u, wind_data_v
        elif model in MODELS.keys():
            return np.load(file_path)[:,:,-2]
        elif model in REGIONAL_MODELS.keys():
            return np.load(file_path)
        else:
            raise ValueError("Model not found")
        
    def load_true_track(self):
        print("Starting to load true track information...")
        previous_location = []
        locations_x = []
        locations_y = []
        true_intensities = []
        windspeeds = []
        time_UTC = []
        intensities = {model: [] for model in (MODELS | BACHELOR_MODEL | REGIONAL_MODELS | PHYSICS_MODELS).keys()} 
        
        for date in self.dates:
            print("For true track, working on date: ", date)
            if len(previous_location) == 0:
                location = self.start_location
            else:
                locations = np.load(f'/fast/proc/cyclones/locations/{date.year}{date.month:02d}/{to_unix(date)}.npz')['x']
                location = self._gather_nearest_location(previous_location, locations)
            
            locations_x.append(location[1] - self.bounding_box['min_x'])
            locations_y.append(location[0] - self.bounding_box['min_y'])
            
            time_UTC.append((date - self.start_time).total_seconds() // 3600)
            
            true_windspeed = self._gather_windspeed_true(location, date)
            windspeeds.append(true_windspeed)
            true_intensities.append(self.convert_wind_to_SSintensity(true_windspeed))
            for model in MODELS.keys():
                data = self.gather_wind_data(model, MODELS[model], date)
                model_windspeed = self.gather_windspeed_model(data, location)
            
                intensities[model].append(np.abs(model_windspeed - true_windspeed))
            
            previous_location = location
        
        self._plot_track(self.track_ax, locations_x, locations_y, ACTUAL_MODEL)
        self._plot_track(self.intensity_track_plts[ACTUAL_MODEL][1], locations_x, locations_y, ACTUAL_MODEL, true_intensities)
        self.intensity_track_plts[ACTUAL_MODEL][0].savefig(f'{self.save_path}/{ACTUAL_MODEL}_intensity_track_plt.png', dpi=300)
        plt.close(self.intensity_track_plts[ACTUAL_MODEL][0])
        
        self.windspeed_at_path_ax.plot(time_UTC, windspeeds, label=ACTUAL_MODEL, marker='o')
        self.intensity_at_location_ax.plot(time_UTC, windspeeds, label=ACTUAL_MODEL, marker='o')
        
        for model in MODELS.keys():
            self.intensity_at_location_ax.plot(time_UTC, intensities[model], label=model, marker='o')
        
    def load_model_tracks(self):
        for model, file_name in MODELS.items():
            print(f"Starting to load model track information for {model}...")
            previous_location = self.start_location
            locations_x = []
            locations_y = []
            intensities = []
            windspeeds = []
            time_UTC = []
            for date in self.dates:
                print(f"For {model}, working on date: ", date)
                wind_data = self.gather_wind_data(model, file_name, date)
                parsed_location = self.parse_location_model(wind_data, previous_location, model, date)
                
                previous_location = parsed_location
                locations_x.append(parsed_location[1] - self.bounding_box['min_x'])
                locations_y.append(parsed_location[0] - self.bounding_box['min_y'])
                time_UTC.append((date - self.start_time).total_seconds() // 3600)
                windspeed = self.gather_windspeed_model(wind_data, parsed_location)
                windspeeds.append(windspeed)
                intensities.append(self.convert_wind_to_SSintensity(windspeed))
                
            self._plot_track(self.track_ax, locations_x, locations_y, model)
            self._plot_track(self.intensity_track_plts[model][1], locations_x, locations_y, model, intensities)
            self.intensity_track_plts[model][0].savefig(f'{self.save_path}/{model}_intensity_track_plt.png', dpi=300)
            plt.close(self.intensity_track_plts[model][0])
            self.windspeed_at_path_ax.plot(time_UTC, windspeeds, label=model, marker='o')
               
    def load_bachelor_tracks(self):
        print("Starting to load bachelor track information...")
        model_name = list(BACHELOR_MODEL.keys())[0]
        previous_location = self.start_location
        locations_x = []
        locations_y = []
        intensities = []
        windspeeds = []
        time_UTC = []
        for date in self.dates:
            print("For bachelor, working on date: ", date)
            wind_data_u, wind_data_v = self.gather_wind_data(list(BACHELOR_MODEL.keys())[0], BACHELOR_MODEL['bachelor'], date)
            
            # For each component, there is a large negative and a large positive
            # Combine each with an average
            location_u, speed_u, raw_location = self.parse_location_bachelor(wind_data_u, previous_location, model_name, date)
            location_v, speed_v, _ = self.parse_location_bachelor(wind_data_v, previous_location, model_name, date, prev_loc=raw_location)
            location = (int((location_u[0] + location_v[0])/2), int((location_u[1] + location_v[1])/2))
            speed = ((speed_u + speed_v)/2) * self.conversion_factor
            
            locations_x.append(location[1] - self.bounding_box['min_x'])
            locations_y.append(location[0] - self.bounding_box['min_y'])
            time_UTC.append((date - self.start_time).total_seconds() // 3600)
            windspeeds.append(speed)
            intensities.append(self.convert_wind_to_SSintensity(speed))
            
            previous_location = location
        
        self._plot_track(self.track_ax, locations_x, locations_y, model_name)
        self._plot_track(self.intensity_track_plts[model_name][1], locations_x, locations_y, model_name, intensities)
        self.intensity_track_plts[model_name][0].savefig(f'{self.save_path}/{model_name}_intensity_track_plt.png', dpi=300)
        plt.close(self.intensity_track_plts[model_name][0])
        self.windspeed_at_path_ax.plot(time_UTC, windspeeds, label=model_name, marker='o') 
              
    def load_physics_tracks(self):
        forecasts = self.storm.get_operational_forecasts()
        for model, model_name in PHYSICS_MODELS.items():
            if model not in forecasts.keys():
                print(f"Physics model {model_name} (key : {model}) not found in forecasts")
                continue
            forecast_times = forecasts[model]
            if self.forecast not in forecast_times.keys():
                print(f"Physics model {model_name} (key : {model}) wasn't forecasts on {self.forecast}")
                continue
            forecast_data = forecast_times[self.forecast]
            print(f"Starting to run physics model {model_name} information...")
        #forecast_data = self.storm.get_nhc_forecast_dict(self.start_time.replace(tzinfo=None))
            latitudes = forecast_data['lat']
            longitudes = forecast_data['lon']
            times = forecast_data['fhr'] 
            
            locations_x = []
            locations_y = []
            windspeeds = []
            intensities = []
            time_UTC = []
            for i, time in enumerate(times):
                forecast_time = self.start_time.replace(tzinfo=None) + timedelta(hours=time)
                if forecast_time > self.end_time.replace(tzinfo=None): break
                
                print(f"For physics model {model_name}, working on date: ", forecast_time)
                row, col = self._convert_lat_lon_to_xy(latitudes[i], longitudes[i])
                vmax = forecast_data['vmax'][i]
                intensity = self.convert_wind_to_SSintensity(vmax)
                
                locations_x.append(col - self.bounding_box['min_x'])
                locations_y.append(row - self.bounding_box['min_y'])
                time_UTC.append(time)
                windspeeds.append(vmax)
                intensities.append(intensity)
            
            self._plot_track(self.track_ax, locations_x, locations_y, model_name)
            self._plot_track(self.intensity_track_plts[model][1], locations_x, locations_y, model_name, intensities)
            self.intensity_track_plts[model][0].savefig(f'{self.save_path}/{model_name}_intensity_track_plt.png', dpi=300)
            plt.close(self.intensity_track_plts[model][0])
            self.windspeed_at_path_ax.plot(time_UTC, windspeeds, label=model_name, marker='o') 
        
    def load_regional_model_tracks(self):
        for model, file_name in REGIONAL_MODELS.items():
            print(f"Starting to load regional model track information for {model}...")
            previous_location = self.start_location
            locations_x = []
            locations_y = []
            intensities = []
            windspeeds = []
            time_UTC = []
            for date in self.dates:
                print(f"For {model}, working on date: ", date)
                data = self.gather_wind_data(model, file_name, date)
                if data.size == 0: continue
                
                windspeed, parsed_location = self.parse_location_regional_model(previous_location, data.squeeze(axis=0))
                
                previous_location = parsed_location
                locations_x.append(parsed_location[1] - self.bounding_box['min_x'])
                locations_y.append(parsed_location[0] - self.bounding_box['min_y'])
                time_UTC.append((date - self.start_time).total_seconds() // 3600)
                windspeeds.append(windspeed)
                intensities.append(self.convert_wind_to_SSintensity(windspeed))
                
            self._plot_track(self.track_ax, locations_x, locations_y, model)
            self._plot_track(self.intensity_track_plts[model][1], locations_x, locations_y, model, intensities)
            self.intensity_track_plts[model][0].savefig(f'{self.save_path}/{model}_intensity_track_plt.png', dpi=300)
            plt.close(self.intensity_track_plts[model][0])
            self.windspeed_at_path_ax.plot(time_UTC, windspeeds, label=model, marker='o')
     
    def save_gif(self, model):
        images = [Image.open(fp) for fp in self.chunk_file_paths[model]]
        images[0].save(f'{self.save_path}/{model}.gif', save_all=True, append_images=images[1:], duration=(1000 / 5), loop=0)
        
    def calculate(self):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(f'{self.save_path}/temp_gif', exist_ok=True)
        
        self.load_true_track()
        self.load_regional_model_tracks()
        self.load_physics_tracks()
        self.load_bachelor_tracks()
        self.load_model_tracks()
        
        self.intensity_at_location_ax.legend()
        self.windspeed_at_path_ax.legend()
        
        self.track_fig.savefig(f'{self.save_path}/track_plt.png', dpi=300)
        
        print("Saving tracking gifs for models that require tracking...")
        for model in (MODELS | BACHELOR_MODEL).keys():
            self.save_gif(model)
            
        self.intensity_at_location_fig.savefig(f'{self.save_path}/intensity_at_location_plt.png', dpi=300)
        self.windspeed_at_path_fig.savefig(f'{self.save_path}/windspeed_at_path_plt.png', dpi=300)
    
# HELPER main functions
def find_forecast_times(forecast):
    forecast_start = D(int(forecast[:4]), int(forecast[4:6]), int(forecast[6:8]), int(forecast[8:]))
    
    min_max = float('inf')
    for model in (MODELS | BACHELOR_MODEL | REGIONAL_MODELS).keys():
        max = float('-inf')
        files = os.listdir('/huge/users/jack/evals/' + 
                   model + '/' + 
                   forecast + '/det')
        
        for file in files:
            if 'intensity' in file: continue
            hour_delta = int(file[:-9])
            if hour_delta > max:
                max = hour_delta
        
        assert max != float('-inf'), "No files found in forecast directory"
        
        if max < min_max:
            min_max = max
            
    assert min_max != float('inf'), "No files found in forecast directory"
    
    forecast_end = forecast_start + timedelta(hours=min_max)
    
    return forecast_start, forecast_end
        
def calculate_bounding_box(start_location, radius, bounding_box_override):
    if bounding_box_override:
        return {
            'min_y': start_location[0] - bounding_box_override[0],
            'max_y': start_location[0] + bounding_box_override[1],
            'min_x': start_location[1] - bounding_box_override[0],
            'max_x': start_location[1] + bounding_box_override[1],
        }
    bounding_box = {
        'min_y': start_location[0] - radius,
        'max_y': start_location[0] + radius,
        'min_x': start_location[1] - radius,
        'max_x': start_location[1] + radius,
    }
    return bounding_box
        
# To use:
# Run run_rt_TC.py for the forecast date
#      Make sure to set which models you want to use in the run_rt_TC.py file config (at the top)
# Check to find the appropriate location using the script at the bottom of this file
# Set the settings to what you want for the storm you're evaluating on and run:
# python3 evals/tc/tc_eval.py
# View results using the script at the end of this file
if __name__ == '__main__':
    bounding_box_override = None
    diff_from_start = timedelta(hours=0)
    diff_from_end = timedelta(hours=0)
    
    # SETTINGS START
    storm_name = 'KATRINA'
    bounding_box_radius = 60
    start_location = [256, 1118]
    forecast = '2005082600'
    # SETTINGS END
    
    # Find bounding box
    bounding_box = calculate_bounding_box(start_location, bounding_box_radius, bounding_box_override)
    
    # Find forecast start and end
    forecast_start, forecast_end = find_forecast_times(forecast)
    
    start_time = forecast_start + diff_from_start # Time we want to start plotting, must be a multiple of 6
    end_time = forecast_end - diff_from_end # Time we want to end plotting
    
    assert start_time < end_time, "Start time must be before end time"
    assert forecast_start <= start_time, "Forecast start must be before start time"
    assert forecast_end >= end_time, "Forecast end must be after end time"
    
    eval = TC_Eval(storm_name, forecast, forecast_start, forecast_end, start_time, end_time, bounding_box, start_location)
    print("Starting evaluation pipeline...")
    eval.calculate()
    eval.remove_temp_files()
    
# SETTINGS Jeanne
'''
storm_name = 'JEANNE'
bounding_box_radius = 80
start_location = [260, 1151]
forecast = '2004092000'
'''
# SETTINGS Jeanne    

# SETTINGS Katrina
'''
storm_name = 'KATRINA'
bounding_box_radius = 60
start_location = [256, 1118]
forecast = '2005082600'
'''
# SETTINGS Katrina  

# SETTINGS Patricia
'''
storm_name = 'PATRICIA'
bounding_box_radius = 60
start_location = [307, 1059]
forecast = '2015102100'
'''
# SETTINGS Patricia

# SETTINGS Ernesto
'''
storm_name = 'ERNESTO'
bounding_box_radius = 60
start_location = [302, 1166]
forecast = '2006082600'
bounding_box_override = [100, 30] 
'''
# SETTINGS Ernesto

# SETTINGS Harvey
'''
storm_name = 'HARVEY'
bounding_box_radius = 60
start_location = [272, 1070]
forecast = '2017082300'
diff_from_start = timedelta(hours=24)
'''
# SETTINGS Harvey

'''
# Initialization part
%cd deep
import sys
sys.path.append('.')
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
earth_background = np.load('/fast/proc/era5/extra/034_sstk/197601/189723600.npz')['x'].squeeze()
threshold = 80

# Discovery part
forecast = '2005082600'
date = D(int(forecast[:4]),int(forecast[4:6]),int(forecast[6:8]),int(forecast[8:]))
location_data = np.load(f'/fast/proc/cyclones/locations/{forecast[:4]}{int(forecast[4:6]):02d}/{to_unix(date)}.npz')['x']
data = np.load(f'/fast/proc/era5/extra/tc-maxws/{forecast[:4]}{int(forecast[4:6]):02d}/{to_unix(date)}.npz')['x']
for location_elem in location_data:
    location = [int(loc_elem) for loc_elem in location_elem]

    plt.imshow(data[location[0] - threshold: location[0] + threshold,
                    location[1] - threshold: location[1] + threshold], zorder=1)
    earth_background_specific = earth_background[location[0] - threshold:location[0] + threshold, 
                                                location[1] - threshold:location[1] + threshold]
    earth_background_specific = np.where(~np.isnan(earth_background_specific), 1, 0)
    plt.imshow(earth_background_specific, zorder=0, cmap=ListedColormap(['gray', 'lightblue']))
    plt.title(location)
    plt.show()
    print(location)
'''    
    
'''
import os
from PIL import Image
import matplotlib.pyplot as plt

storm_name = 'PATRICIA'
forecast = '2015102100'

path = f'/huge/users/jack/evals/eval_plots/{storm_name}/{forecast}'
image_paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.png')]

for img_path in image_paths:
    try: image = Image.open(img_path)
    except Exception as e: continue
    plt.imshow(image)
    plt.axis('off')
    plt.show()
'''  