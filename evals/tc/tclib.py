from datetime import datetime
from utils import *
import pytz
import re


#hur_info = {
#    'ian': {'stormId': '2022266N12294', 'id':'al09','bbox': [35, -90, 8, -50], 'landfall': datetime(2022, 9, 28, 19, 10)},
#    'fiona': {'stormId': '2022257N16312', 'id':'al07','bbox': [45, -90, 12, -30],},
#}

storms_2024 = {
  "beryl": (datetime(2024, 6, 28), datetime(2024, 7, 9)),
  "debby": (datetime(2024, 8, 3), datetime(2024, 8, 9)),
  "ernesto": (datetime(2024, 8, 12), datetime(2024, 8, 20)),
  "francine": (datetime(2024, 9, 9), datetime(2024, 9, 12)),
  "helene": (datetime(2024, 9, 24), datetime(2024, 9, 27)),
  "kirk": (datetime(2024, 9, 29), datetime(2024, 10, 7)),
  "milton": (datetime(2024, 10, 5), datetime(2024, 10, 10)),
  "oscar": (datetime(2024, 10, 19), datetime(2024, 10, 22)),
}

def get_storm_year(storm):
    if storm in storms_2024:
        return 2024
    else:
        return 2022

import tropycal.tracks as tracks

track_db = None
CACHEPATH = 'evals/tc/ignored/'
def get_trackdb():
    global track_db
    track_db = basin = tracks.TrackDataset(basin='north_atlantic',source='hurdat',include_btk=True)
    return track_db
    if not os.path.exists(f'{CACHEPATH}tracks_ibtracs.pkl'):
        os.makedirs(CACHEPATH,exist_ok=True)
        db = tracks.TrackDataset(basin='all',source='ibtracs')
        with open(f'{CACHEPATH}/tracks_ibtracs.pkl','wb') as f:
            pickle.dump(db,f)
    else:
        print("loading from cache: ", f'{CACHEPATH}/tracks_ibtracs.pkl')
        with open(f'{CACHEPATH}/tracks_ibtracs.pkl','rb') as f:
            db = pickle.load(f)
    return db


storm_cache = {}
def get_storm(name,year=None):
    global track_db, storm_cache
    if name in storm_cache:
        return storm_cache[name]
    if track_db is None:
        track_db = get_trackdb()
    if year is None:
        year = get_storm_year(name)
    storm = track_db.get_storm((name,year))
    storm_cache[name] = storm
    return storm

def get_storms(names: list[str]):
    basin = tracks.TrackDataset(basin='north_atlantic',source='hurdat',include_btk=True)
    return [basin.get_storm((name, get_storm_year(name))) for name in names]

def get_hur_bbox(name):
    storm = get_storm(name)
    padlon = 4
    padlat = 2
    bbox = [max(storm.lat)+padlat, min(storm.lon)-padlon, min(storm.lat)-padlat, max(storm.lon)+padlon]
    return bbox
    #match name:
    #    case 'ian': return [45, -100, 8, -50]
    #    case 'fiona': return [45, -90, 12, -30]
    #    case _: assert False

def parse_advisory_time(time_str):
    # ex ''800 PM EDT Fri Sep 23 2022'
    # return a datetime object, converted to utc time

    # Example mapping (extend this as needed)
    tz_map = {
        'EST': 'America/New_York',
        'EDT': 'America/New_York',
        'CST': 'America/Chicago',
        'AST': 'America/Puerto_Rico',
        # Add more mappings
    }

    s = '200 AM EST Thu Sep 15 2022'
    tz_match = re.search(r'\b(AST|EST|EDT|CST)\b', s)
    if tz_match:
        tz_abbr = tz_match.group(0)
        cleaned = re.sub(r'\b(AST|EST|EDT|CST)\b', '', s).strip()
        dt_naive = datetime.strptime(cleaned, '%I00 %p  %a %b %d %Y')
        
        if tz_abbr in tz_map:
            tz = pytz.timezone(tz_map[tz_abbr])
            dt_localized = tz.localize(dt_naive)
        else:
            assert False, f'No timezone mapping for {tz_abbr}'

    utc_dt = dt_localized.astimezone(timezone.utc)
    return utc_dt


def get_hur_track(name):
    storm = get_storm(name)
    lats_lons_time = [(storm.lat[i], storm.lon[i], to_unix(storm.time[i])) for i in range(len(storm.lat))]
    return lats_lons_time

def get_hur_tracks(names: list[str]):
    storms = get_storms(names)
    tracks = {}
    for storm in storms:
        lats_lons_time = [(storm.lat[i], storm.lon[i], to_unix(storm.time[i])) for i in range(len(storm.lat))]
        tracks[storm.name] = lats_lons_time
    return tracks

def get_hur_loc(track, nix):
    # Ensure the track is sorted by time
    
    # Find points between which nix falls
    for i in range(len(track) - 1):
        if track[i][2] <= nix <= track[i + 1][2]:
            lat1, lon1, time1 = track[i]
            lat2, lon2, time2 = track[i + 1]
            
            # Linear interpolation
            if time2 != time1:  # Avoid division by zero
                ratio = (nix - time1) / (time2 - time1)
                lat_nix = lat1 + (lat2 - lat1) * ratio
                lon_nix = lon1 + (lon2 - lon1) * ratio
                return lat_nix, lon_nix

    return None  # Return None if nix is outside the range of available data

from math import radians, cos, sin, sqrt, atan2

def haver_dist(p1,p2):
    lat1, lon1 = p1
    lat2, lon2 = p2

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Radius of Earth in kilometers. Use 3956 for miles
    r = 6371.0
    
    return r * c

def get_pred_track(hurname,forecast_at):
    f_at = date_str2date(forecast_at)
    storm = get_storm(hurname)
    fcst = storm.get_nhc_forecast_dict(f_at)
    times = [to_unix(fcst['init'] + timedelta(hours=h)) for h in fcst['fhr']]
    pred_track = [(fcst['lat'][i], fcst['lon'][i], times[i]) for i in range(len(fcst['lat']))]    
    return pred_track


def get_gfs_pred_track(hurname,forecast_at):
    f_at = date_str2date(forecast_at)
    storm = get_storm(hurname)
    try: storm.forecast_dict
    except: storm.get_operational_forecasts()
    check_keys = ["AVNI","AVNO","AVNX"]
    #print(forecast_at)
    fcst = None
    for k in check_keys:
        if k in storm.forecast_dict:
            fcst_ = storm.forecast_dict[k]
            #print(list(fcst_.keys()))
            if forecast_at in fcst_:
                print(f"found {k} forecast at {forecast_at}")
                fcst = fcst_[forecast_at]
                break
    if fcst is None:
        print(f"no forecast found for {hurname} at {forecast_at}")
        return None 
    times = [to_unix(fcst['init'] + timedelta(hours=h)) for h in fcst['fhr']]
    pred_track = [(fcst['lat'][i], fcst['lon'][i], times[i]) for i in range(len(fcst['lat']))]    
    return pred_track

def get_big_storms():
    big_storms = {}
    for id,storm in get_trackdb().get_season(year=2022,basin='all').dict.items():
        times = sorted(list(set([get_date_str(x.replace(hour=0, minute=0, second=0, microsecond=0)) for x in storm['time']])))
        if times[0] < '20220826':
            continue
        if np.max(storm['vmax']) < 100:
            continue
        if storm['name'] == 'UNNAMED':
            continue
        big_storms[storm['name'].lower()] = times
    return big_storms

if __name__ == '__main__':
    storm = get_storm('hinnamnor')
    print(storm)
    storm.plot_models(datetime(2022,9,2))
    plt.savefig('ignored/ohp2.png')
    pass