import tarfile
import multiprocessing
import pickle
import numpy as np
import sys
from datetime import datetime, timedelta
from tqdm import tqdm
import zipfile
import io

def svpw(t):
    #t = t_ + 273.15
    #t /= 1000
    h = [-0.58002206e4, 0.13914993e1, -0.48640239e-1, 0.41764768e-4, -0.14452093e-7, 0.65459673e1]
    r = 0
    for i in range(-1, 4):
        r += h[i+1] * t**i
    r += h[-1] * np.log(t)
    return np.exp(r)

def svpi(t):
    #t = t_ + 273.15
    h = [-0.56745359e4, 0.63925247e1, -0.96778430e-2, 0.62215701e-6, 0.20747825e-8, -0.94840240e-12, 0.41635019e1]
    r = 0
    for i in range(0, 6):
        r += h[i] * t**(i-1)
    r += h[-1] * np.log(t)
    return np.exp(r)

def svp(t_, sw=None):
    if sw is None: sw = t_
    if sw >= 273.15: return svpw(t_)
    else: return svpi(t_)


ss = open('/huge/igra/igra2-station-list.txt').read().split("\n")[:-1]
stations = {}
for s in ss:
    sp = s.split()
    stations[sp[0]] = (float(sp[1]), float(sp[2]), float(sp[3]), int(sp[-3]), int(sp[-2]))

def parse_header(l):
    sp = l.split()
    #print("parsing", l)
    lat = float(l[55:62])/10000.
    lon = float(l[63:71])/10000.
    reflat, reflon, refelev, _, _ = stations[sp[0][1:]]
    if abs(reflat - lat) > 0.01 or abs(reflon - lon) > 0.01:
        #print("uhhhhhhhhhhhH", reflat, lat, reflon, lon)
        return
    date = datetime(int(l[13:17]), int(l[18:20]), int(l[21:23]), int(l[24:26]))
    rel = l[27:31]
    hh = rel[0:2]
    mm = rel[2:4]
    if hh != '99' and mm != '99':
        corr = date - timedelta(hours=24 if hh >= '13' else 0)
        corr = corr.replace(hour=int(hh), minute=int(mm))
        delta = abs((date-corr).total_seconds())
        if delta >= 3600 * 2:
            return
        return (reflat, reflon, refelev, corr)
    return

fn = "IGRA_v2.2_data-por_s19050404_e20241117_c20241118_part%dof4.tar"
for number in range(1, 5):
    tar = tarfile.open(fn%number)
    sz = 0
    members = [name for name in tar.getnames()]
    tar.close()
    def op(name):
        print("Processing", name)
        si = stations[name.split('-')[0]]
        if si[-1] < 1985:
            #print("skipping station", name, si)
            return
        tar = tarfile.open(fn%number)
        f = tar.extractfile(name)
        txt = zipfile.ZipFile(f)
        with io.TextIOWrapper(txt.open(txt.namelist()[0])) as txt:

            for line in txt:
                if not line.startswith("#"): continue
                sp = line.split()
                if sp[1] <= '1980' or line[29:31] == '99': continue
                else: break

            try: hdr = parse_header(line)
            except:
                #print("parse failure on header", line, name)
                hdr = None
                return
            data = []
            Data = []
            for line in txt:
                if line.startswith("#"):
                    # finish off
                    if hdr is not None:
                        aaa = []
                        for l in data:
                            #sys.stdout.write(l)
                            typ = l[0:2]
                            if typ == '21': is_sfc = True
                            else: is_sfc = False
                            etime = l[3:8]
                            if etime[0] == '-':
                                etime = np.nan
                            else:
                                try: mins = int(etime[0:3])
                                except: mins = 0
                                etime = mins*60 + int(etime[3:5])
                            pflag = l[15]
                            if pflag == 'B' or l[9] != '-':
                                pres = int(l[9:15])
                            else:
                                pres = np.nan
                            zflag = l[21]
                            if zflag == 'B' or is_sfc or l[16] != '-':
                                gph = int(l[16:21])
                            else:
                                gph = np.nan
                            tflag = l[27]
                            if tflag == 'B':
                                temp = float(l[22:27])*0.1 + 273.15
                            else:
                                temp = np.nan
                            rh = l[28:33]
                            if rh == '-8888' or rh == '-9999':
                                rh = np.nan
                            else:
                                rh = float(rh) * 0.1
                            dpd = l[34:39]
                            if dpd == '-8888' or dpd == '-9999':
                                dpd = np.nan
                            else:
                                dpd = float(dpd) * 0.1

                            wdir = l[46:51]
                            if wdir == '-8888' or wdir == '-9999':
                                wdir = np.nan
                            else:
                                wdir = float(wdir)

                            wspd = l[46:51]
                            if wspd == '-8888' or wspd == '-9999':
                                wspd = np.nan
                            else:
                                wspd = float(wspd) * 0.1
                            if not np.isnan(wdir) and not np.isnan(wspd):
                                ucomp = -wspd * np.sin(wdir * np.pi/180)
                                vcomp = -wspd * np.cos(wdir * np.pi/180)
                            else:
                                ucomp = np.nan
                                vcomp = np.nan
                            if np.isnan(rh) and not np.isnan(dpd) and not np.isnan(temp):
                                dp = temp - dpd
                                #rh = svp(dp)/svp(temp/10.) * 100
                                rh = svpw(dp)/svpw(temp) * 100

                            aaa.append((typ, etime, pres, gph, temp, rh, ucomp, vcomp))
                        aaa = np.array(aaa, dtype=np.float32)
                        Data.append((hdr, aaa))
                    data = []
                    hdr = parse_header(line)
                else:
                    data.append(line)
        tar.close()
        return Data



    if 0:
        p = multiprocessing.Pool(48)
        proc = list(tqdm(p.imap(op, members), total=len(members)))
        with open("pick%d.pickle" % number, "wb") as f:
            pickle.dump(proc, f)
        del proc

    else: 
        for m in members:
            op(m)
