import sys
sys.path.append('..')
from utils import *

print=builtins.print

fmt = "https://data.ecmwf.int/forecasts/%04d%02d%02d/%02dz/0p4-beta/enfo/%04d%02d%02d%02d0000-%dh-enfo-ef.%s"


utc = datetime.utcnow()
utc -= timedelta(minutes=utc.minute, seconds=utc.second, microseconds=utc.microsecond)
utc -= timedelta(hours=utc.hour%12)

dates = get_dates([(datetime(2023,12,16,12), datetime(2023,12,19,12), timedelta(hours=6))])
dates = get_dates([(datetime(2023,12,19,18), datetime(2023,12,22,0), timedelta(hours=6))])
dates = get_dates([(datetime(2023,12,24,12), datetime(2023,12,27,12), timedelta(hours=6))])
dates = get_dates([(datetime(2024,1,4,12), utc, timedelta(hours=12))])

urls = []

for date in dates:
    for fct in [0,24,72,120,168,240]:
        for w in ["grib2", "index"]:
            base = "/slow/ens_free/%04d%02d" % (date.year, date.month)
            os.makedirs(base, exist_ok=True)
            url = fmt % (date.year, date.month, date.day, date.hour, date.year, date.month, date.day, date.hour, fct, w)
            outf = url.split("/")[-1]
            if not os.path.exists(base+"/"+outf):
                print(url+"\n\tout=%04d%02d/%s"%(date.year, date.month, outf))
