import os
import matplotlib.pyplot as plt
import pygrib
import sys
sys.path.append('..')
from utils import *
import pickle
with open("/fast/consts/normalization.pickle", "rb") as f:
    norm = pickle.load(f)

levels = levels_tiny

levels_file = [10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 925, 950, 1000]
levels = levels_file

pvarlist = ["Geopotential", "Temperature", "U component of wind", "V component of wind", "Specific humidity"]
svarlist = ["10 metre U wind component", "10 metre V wind component", "2 metre temperature", "Mean sea level pressure"]

base = "/fast/ignored/mars_data/"
#ls = sorted([x for x in os.listdir(base) if x.startswith("output618_") and x.endswith(".grib") and x >= "output618_2023070906.grib"])
ls = sorted([x for x in os.listdir(base) if x.startswith("output_") and x.endswith(".grib")])
ls = sorted([x for x in os.listdir(base) if x.startswith("output_") and x.endswith(".grib") and x >= "output_20240114"])
ls = sorted([x for x in os.listdir(base) if x.startswith("oldoutput_") and x.endswith(".grib")])
ls = sorted([x for x in os.listdir(base) if x.startswith("oldoutput_") and x.endswith(".grib") and x >= "oldoutput_2021082612"])
ls = sorted([x for x in os.listdir(base) if x.startswith("oldoutput_") and x.endswith(".grib") and x >= "oldoutput_2022072812"])
bad = 0
for fn in ls:
    print("------------ file", fn, "------------------")
    f = pygrib.open(base+fn)
    i = 0
    l = len(f)
    pr = np.zeros((720, 1440, 5, len(levels)), dtype=np.float16)+np.nan
    sfc = np.zeros((720, 1440, 4), dtype=np.float16)+np.nan
    for grb in f:
        #print(grb)
        if grb.name in pvarlist and grb.level in levels:
            vals = grb.values[:]
            idxvar = pvarlist.index(grb.name)
            nn = norm[core_pressure_vars[idxvar]]
            whlev = levels_full.index(grb.level)
            normed = (vals - nn[0][whlev])/np.sqrt(nn[1][whlev])
            pr[:, :, idxvar, levels.index(grb.level)] = normed[:720]
        if grb.name in svarlist:
            idxvar = svarlist.index(grb.name)
            nn = norm[core_sfc_vars[idxvar]]
            #print(grb.name, idxvar, nn[0][0], np.sqrt(nn[1][0]))
            normed = (grb.values[:]- nn[0][0])/np.sqrt(nn[1][0])
            sfc[:, :, idxvar] = normed[:720]

        i += 1
        if i == len(levels_file)*len(pvarlist) + len(svarlist):
            d = grb.validDate
            print("made it to the end, should have everything", d, "total bad", bad)
            bbase = "/fast/proc/neohres/f000/%04d%02d/" % (d.year, d.month)
            os.makedirs(bbase, exist_ok=True)
            outpath = "/fast/proc/neohres/f000/%04d%02d/%d.npz" % (d.year, d.month, (d-datetime(1970,1,1)).total_seconds())
            era5 = "/fast/proc/era5/f000/%04d%02d/%d.npz" % (d.year, d.month, (d-datetime(1970,1,1)).total_seconds())
            if os.path.exists(outpath):
                try:
                    print("ref exists, comparing and not overwriting!!")
                    ref = np.load(outpath)
                    era5 = np.load(era5)
                    #plt.imshow(delta[:,:,0])
                    #plt.savefig("/fast/public_html/delta.png", bbox_inches='tight')
                    #print("rms pr", np.sqrt(np.mean(np.square(pr.astype(np.float32)-ref["pr"].astype(np.float32)), axis=(0,1))))
                    print("rms sfc", np.sqrt(np.mean(np.square(sfc.astype(np.float32)-ref["sfc"][:720].astype(np.float32)), axis=(0,1))))
                    print("era5 new", np.sqrt(np.mean(np.square(sfc.astype(np.float32)-era5["sfc"].astype(np.float32)), axis=(0,1))))
                    print("era5 old", np.sqrt(np.mean(np.square(ref["sfc"][:720].astype(np.float32)-era5["sfc"].astype(np.float32)), axis=(0,1))))
                    """
                    print("biasnew", np.mean(sfc.astype(np.float32), axis=(0,1)))
                    print("stdnew", np.std(sfc.astype(np.float32), axis=(0,1)))

                    print("biasold", np.mean(ref["sfc"].astype(np.float32), axis=(0,1)))
                    print("stdold", np.std(ref["sfc"].astype(np.float32), axis=(0,1)))
                    """
                except:
                    print("comparison failed")
                    traceback.print_exc()
            isbad = ((~np.isfinite(pr)).sum() + (~np.isfinite(sfc)).sum()) > 0
            if isbad:
                print("date", grb.validDate, "is bad!!!", fn, np.isfinite(pr).sum(), np.isfinite(sfc).sum())
                bad += 1
            op2 = outpath.replace(".npz", ".tmp.npz")
            np.savez(op2,pr=pr,sfc=sfc)
            os.rename(op2, outpath)
            i = 0
            pr += np.nan
            sfc += np.nan
    #print("going through", f)
    f.close()

