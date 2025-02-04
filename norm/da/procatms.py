from __future__ import print_function
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import ncepbufr
import numpy as np
import sys
from datetime import datetime, timedelta

satname = sys.argv[1]
date = sys.argv[2]

_dat = datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 23)
_base = "/fast/proc/satobs/%s/%04d%02d/%d.npz" % (satname, _dat.year, _dat.month, (_dat-datetime(1970,1,1)).total_seconds() )
if os.path.exists(_base):
    print("kk done", _base)
    exit()

hdrstr ='YEAR MNTH DAYS HOUR MINU PCCF ELRC SAID PTID GEODU'

# read gpsro file.

mean_amua, std_amua = [208.11433, 200.07397, 239.66342, 252.61078, 246.26218, 230.94685, 222.31076, 216.79262, 210.95709, 214.30807, 220.71764, 228.75981, 240.04573, 251.177, 238.29852], [40.338806, 41.497913, 19.817978, 13.5695715, 11.055716, 8.138229, 7.369916, 8.368133, 10.854924, 11.318849, 12.498083, 13.425758, 13.70429, 12.700977, 27.190237]

mean_atms, std_atms = [209.518865, 201.116778, 239.50126 , 249.30659 , 252.9082  , 246.775408, 231.462015, 222.500808, 217.218617, 211.59793 , 215.116865, 221.556385, 228.964267, 241.159455, 250.80906 , 237.884928, 258.607035, 260.176345, 258.14215 , 255.904132, 252.182983, 246.94686 ], [41.237975, 41.979399, 21.9403  , 17.243942, 13.995348, 11.227189, 8.208647,  7.063055,  8.130123, 10.969226, 11.324154, 12.453338, 13.527296, 14.045363, 13.322912, 28.550902, 29.529412, 21.900062, 17.83456 , 14.576401, 11.84562 , 10.112091]
mean_atms = np.array(mean_atms, dtype=np.float32)
std_atms = np.array(std_atms, dtype=np.float32)

mean_amua = np.array(mean_amua, dtype=np.float32)
std_amua = np.array(std_amua, dtype=np.float32)

if satname == "atms":
    mean = mean_atms
    std = std_atms
elif satname == "1bamua":
    mean = mean_amua
    std = std_amua
else:
    assert False


dd = date[:6]
tar = "/huge/proc/ncarsat/%s/%s.%s.tar.gz" % (dd, satname, date)

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

with temp_tar(tar) as tmp:
    ls = sorted(os.listdir(tmp))
    if len(ls) == 1:
        ls = [ls[0] + '/' + x for x in os.listdir(tmp+'/'+ls[0])]
    for ff in ls:
        bufr = ncepbufr.open(os.path.join(tmp, ff))
        #bufr.print_table()
        from pprint import pprint
        #pprint(bufr.__dict__)
        lats = []
        lons = []
        cyc = datetime(2024,11,24)
        dts = []
        saids = set()
        fovns = set()
        sozas = []
        sazas = []
        Data = defaultdict(list)

        def saveit(dat, xx):
            xx = np.array(xx).astype(np.float16)
            base = "/fast/proc/da/%s/%04d%02d/" % (satname, dat.year, dat.month)
            os.makedirs(base, exist_ok=True)
            fn = "%d.npz" % (dat - datetime(1970,1,1)).total_seconds()
            tm = base + fn.replace(".npz", ".tmp.npz")
            np.savez_compressed(tm, x=xx)
            os.rename(tm, base+fn)
            #Data = []

        last_dat = None
        old = None
        while bufr.advance() == 0:
            #if len(Data) > 100000:
            #    break
            while bufr.load_subset() == 0:
                hdr = bufr.read_subset('SAID FOVN YEAR MNTH DAYS HOUR MINU SECO CLAT CLON SAZA SOZA HMSL LSQL SOLAZI').squeeze()
                hdr2 = bufr.read_subset('CLATH CLONH').squeeze()
                if hdr2.shape == (2,0):
                    lat = hdr[8]
                    lon = hdr[9]
                else:
                    lat = hdr2[0]
                    lon = hdr2[1]
                saza = hdr[10]
                soza = hdr[11]
                sazas.append(saza)
                sozas.append(soza)
                #data = bufr.read_subset('TMBR').squeeze()
                data = bufr.read_subset('TMBR', rep=True, seq=False, events=False).squeeze().filled(np.nan)
                data = (data - mean)/std
                data = data.astype(np.float16)
                if int(hdr[7]) == 60:
                    hdr[7] = 59
                d = datetime(int(hdr[2]), int(hdr[3]), int(hdr[4]), int(hdr[5]), int(hdr[6]), int(hdr[7]))
                """
                if old is not None:
                    if d < old:
                        print("uhhh", old, d, hdr)
                old = d
                """
                nxt_hour = d.replace(minute=0, second=0)
                nxt_hour += timedelta(hours=1)
                dt = (d - nxt_hour).total_seconds()/3600
                said = hdr[0]
                ex = [dt, said, lat * np.pi/180, lon*np.pi/180, saza*np.pi/180, soza*np.pi/180]
                data = np.concatenate([ex, data]).astype(np.float16)
                #print(nxt_hour, last_dat)
                """
                if last_dat is not None and nxt_hour != last_dat:
                    print("saving!", last_dat, nxt_hour)
                    saveit(last_dat)
                    assert len(Data) == 0
                """

                Data[nxt_hour].append(data)
                #print("aa", data, data.shape)
                saids.add(hdr[0])
                #fovns.add(hdr[1])
                lats.append(hdr2[0])
                lons.append(hdr2[1])
                d = datetime(int(hdr[2]), int(hdr[3]), int(hdr[4]), int(hdr[5]), int(hdr[6]), int(hdr[7]))
                #print(d, hdr[8], hdr2[0], hdr[9], hdr2[1])
                dts.append((d - cyc).total_seconds()/3600)
                last_dat = nxt_hour
                #print(hdr, hdr2)
                continue
                yyyymmddhh ='%04i%02i%02i%02i%02i' % tuple(hdr[0:5])
        #print("last one was", nxt_hour)
        #saveit(nxt_hour)
        for hr in Data:
            saveit(hr, Data[hr])
        bufr.close()
        #print("yoo", Data.shape, Data.dtype)

exit()
np.savez("test.npz", x=Data)
np.savez_compressed("testcomp.npz", x=Data)
bufr.close()

plt.scatter(lons, lats, c=np.arange(len(lons)), s=0.5)
plt.savefig("/fast/public_html/atms.png", dpi=300, bbox_inches='tight')
exit()

#import pdb; pdb.set_trace()

"""
import matplotlib.pyplot as plt
plt.hist(dts, bins=100)
plt.show()
import pdb; pdb.set_trace()
"""

import matplotlib.pyplot as plt
plt.scatter(lons, lats)
plt.show()
"""
summer 12
[207.41145 199.61728 239.91637 250.08658 253.5076  247.11443 231.68227
 221.89772 215.89336 211.08928 213.5152  218.34415 226.1631  238.40623
 248.1834  239.88872 263.29245 262.41608 259.0964  256.19476 252.37082
 247.19125] [43.45228   44.872543  24.93122   19.815186  15.598408  11.9362135
  8.596519   7.640989   8.290658  10.68328   11.833132  14.099226
 15.9476595 16.394783  14.604221  30.094633  29.192747  23.748135
 20.505363  17.32743   14.045709  11.794689 ]

summer 18
[216.52931 208.07706 242.76868 250.75829 253.36566 246.85565 231.72934
 221.53545 215.51099 210.62122 213.47289 218.88646 226.08228 239.00739
 249.0581  242.7847  261.73944 260.4542  257.50342 254.95296 250.99272
 245.7867 ] [43.317905 44.72014  24.139668 19.03076  15.173881 12.061937  9.270667
  8.414004  8.955522 11.194419 12.35491  14.521682 16.140802 16.245092
 13.980749 29.818382 28.300152 22.167078 18.629385 15.252735 12.012953
 10.015164]

winter 00
[205.6968  196.46492 237.31424 248.20079 252.94226 247.23445 231.44518
 223.48502 219.08316 212.98148 217.27213 224.87129 232.21341 244.02225
 253.11314 234.14017 255.47772 259.813   258.46243 256.57898 253.07819
 247.65634] [37.354694  37.297104  17.314322  13.283131  11.487693   9.947912
  7.1802373  5.633362   7.2422323 11.016128  10.180655   9.164983
  9.48702   10.873287  12.116337  25.108849  29.187067  19.867031
 15.113177  11.979009  10.10644    8.968632 ]

winter 06
[208.4379  200.30785 238.00575 248.1807  251.81728 245.8971  230.99127
 223.08504 218.38696 211.69974 216.20724 224.12364 231.39828 243.20195
 252.8816  234.72612 253.91853 258.0221  257.50635 255.88983 252.2902
 247.15315] [39.661106  39.643177  20.088902  15.904555  13.273853  10.77827
  7.6016083  6.0022006  7.3140574 10.833091  10.2728405  9.573496
  9.7190695 10.657843  11.614025  27.98776   30.321692  21.413366
 16.56891   13.117111  10.736706   9.338701 ]


"""
