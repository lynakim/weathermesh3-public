#!/usr/bin/env python
""" 
Python script to download selected files from rda.ucar.edu.
After you save the file, don't forget to make it executable
i.e. - "chmod 755 <name_of_script>"
"""
import sys, os
from urllib.request import build_opener

opener = build_opener()

tmp = 'https://data.rda.ucar.edu/ds084.1/%04d/%04d%02d%02d/gfs.0p25.%04d%02d%02d00.f024.grib2'

from datetime import datetime, timedelta

d = datetime(2017, 1, 1)


filelist = [
]

while d <= datetime(2017, 12, 1):
    filelist.append(tmp % (d.year, d.year, d.month, d.day, d.year, d.month, d.day))
    d += timedelta(days=11)

print(filelist)
filelist = filelist[1:]

for file in filelist:
    ofile = "/fast/ignored/gfsnorm/"+os.path.basename(file)
    sys.stdout.write("downloading " + ofile + " ... ")
    sys.stdout.flush()
    infile = opener.open(file)
    outfile = open(ofile, "wb")
    outfile.write(infile.read())
    outfile.close()
    sys.stdout.write("done\n")
