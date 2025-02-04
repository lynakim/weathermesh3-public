import cdsapi
import sys
import os

year = int(sys.argv[1])

c = cdsapi.Client()

result = c.retrieve(
    'insitu-observations-surface-land',
    {
        'format': 'zip',
        'time_aggregation': 'sub_daily',
        'variable': [
            'air_pressure', 'air_pressure_at_sea_level', 'air_temperature',
            'dew_point_temperature', 'wind_from_direction', 'wind_speed',
        ],
        'usage_restrictions': [
            'restricted', 'unrestricted',
        ],
        'data_quality': 'passed',
        'year': str(year),
        'month': ['%02d'%m for m in range(1,13)],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
    })
url = result.location
print("yooo url is", url)
#,
#    'obs%d.zip' % year)
os.system('cd /fast/ignored/hres/raw && aria2c -j16 -x16 -s16 %s -o raw_%d.zip' % (url, year))
