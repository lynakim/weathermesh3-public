import urllib.request
from tqdm import tqdm

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

for year in range(1979,2024):
    print(year)
    url = f"https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/archive/ghcn-hourly_v1.0.0_d{year}_c20240301.tar.gz"
    filename = f"raw/ghcn-hourly_v1.0.0_d{year}_c20240301.tar.gz"

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename, reporthook=t.update_to)