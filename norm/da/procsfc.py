import sys
import os
from collections import defaultdict
import time
import re
from pprint import pprint
import numpy as np
from datetime import datetime, timedelta

date = sys.argv[1]

_base = "/fast/proc/da/sfc/%s/%s.npy" % (date[:6], date[:8]+'21')
if os.path.exists(_base):
    print("kk done", _base)
    exit()


hdr = "   REC      OBS       REPORTTIME   STATION   LAT   LON   ELEV   STNPR  STNDSLP  ALTIM     AIR.T    DEWPT    R.HUM    WINDDIR     WINDSPD      HOR     3HPR   24HPR"
hdr = hdr.split()
#hdr = [x.lstrip().rstrip() for x in hdr]

def tr(l):
    o = {}
    for h, x in zip(hdr, l):
        if h == "REPORTTIME":
            d = datetime(int(x[0:4]), int(x[4:6]), int(x[6:8]), int(x[8:10]), int(x[10:12]))
            o[h] = d
        elif '9999' in x: o[h] = np.nan

        else:
            try: o[h] = float(x)
            except: o[h] = x
    return o

dd = date[:6]
tar = "/huge/proc/ncarsat/%s/adpsfc.%s.tar.gz" % (dd, date)

import tarfile
import tempfile
from contextlib import contextmanager

@contextmanager
def temp_tar(tar_path):
    temp_dir = tempfile.mkdtemp()
    try:
        print("extracting into", temp_dir, tar_path)
        tarfile.open(tar_path).extractall(temp_dir)
        yield temp_dir
    finally:
        import shutil
        shutil.rmtree(temp_dir)


outfiles = defaultdict(list)

with temp_tar(tar) as tmp:
    ls = sorted(os.listdir(tmp))
    if len(ls) == 1:
        ls = [ls[0] + '/' + x for x in os.listdir(tmp+'/'+ls[0])]
    for ff in ls:
        bufr = os.path.join(tmp, ff)
        out = os.path.join(tmp, "out.txt")
        if os.path.exists(out):
            os.unlink(out)
        cmd = "/home/windborne/rda-bufr-decode-ADPsfc/exe/bufrsurface.x %s %s /home/windborne/rda-bufr-decode-ADPsfc/configs/bufrsurface_config_all" % (bufr, out)
        os.system(cmd)
        with open(out) as ff:
            txt = ff.read()

        lines = txt.split("|")[2:-1]
        lines = [x.split() for x in lines]
        lens = [len(x) for x in lines]
        assert len(set(lens)) == 1
        assert lens[0] == 18
        assert len(hdr) == 18
        lines = [tr(x) for x in lines]

        cols = ["is_ship", "reltime_hours", "lat_deg", "lon_deg", "elev_m", "unnorm_mslp", "unnorm_2t", "unnorm_2d", "unnorm_10u", "unnorm_10v", "unnorm_viz_m", "unnorm_3hpr_mm", "unnorm_24hpr_mm"]
        for dic in lines:
            row = []
            row.append(dic['REC'] == "SFCSHP")
            d = dic['REPORTTIME']
            if d.minute != 0 or 1: # actually, always go to the next hour
                nexthour = d + timedelta(minutes=60-d.minute)
            else: nexthour = d

            reltime_hours = (d - nexthour).total_seconds()/3600
            assert -1<=reltime_hours<=0
            #print(reltime_hours, d, nexthour)

            if dic['LAT'] > 90 or dic['LAT'] < -90 or dic['LON'] < -180:
                print("Bad point!!!")
                pprint(dic)
                continue
            if dic['LON'] > 180: dic['LON'] -= 360

            row.append(reltime_hours)
            row.append(dic['LAT'])
            row.append(dic['LON'])
            row.append(dic['ELEV'])
            row.append(dic['STNDSLP'])
            row.append(dic['AIR.T'])
            row.append(dic['DEWPT'])

            wspd = dic['WINDSPD']
            wdir = dic['WINDDIR']

            if not np.isnan(wspd) and not np.isnan(wdir):
                ucomp = -wspd * np.sin(wdir * np.pi/180)
                vcomp = -wspd * np.cos(wdir * np.pi/180)

            row.append(ucomp)
            row.append(vcomp)

            row.append(dic['HOR'])
            row.append(dic['3HPR'])
            row.append(dic['24HPR'])

            assert len(row) == len(cols)

            outfiles[nexthour].append(row)

for k in sorted(outfiles.keys()):
    a = np.array(outfiles[k]).astype(np.float32)
    #nn = np.sum(np.isnan(a),axis=0)/a.shape[0]
    #pprint(list(zip(cols, nn)))
    base = "/fast/proc/da/sfc/%04d%02d/"%(k.year, k.month)
    os.makedirs(base, exist_ok=True)
    out = base + "%04d%02d%02d%02d.npy" % (k.year, k.month, k.day, k.hour)
    tmp = out.replace(".npy", ".tmp.npy")
    np.save(tmp, a)
    os.rename(tmp, out)
    print("saved", out, os.path.getsize(out)/1e6, "MB", "ship%", np.mean(a[:,0]))

#pprint(lines)
