import sys
from pprint import pprint
import re

year = sys.argv[1]

import requests

base = 'https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.061/%s.01.01/' % year
ls = requests.get(base).content

files = re.findall('a href="(.*?hdf)"', ls.decode('ascii'))

for f in files:
    print(base + f)

# wget -P /fast/ignored/modis/2020 --user jcreus --password 'Ihmlawtd19!!' --auth-no-challenge -i modis2020