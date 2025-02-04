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

levels_file = [10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 925, 950, 100]

pvarlist = ["Geopotential", "Temperature", "U component of wind", "V component of wind", "Specific humidity"]
svarlist = ["10 metre U wind component", "10 metre V wind component", "2 metre temperature", "Mean sea level pressure"]

# fuuuuck all this shit see below

mp = {"Sea surface temperature": "034_sstk",
        "Large-scale precipitation": "142_lsp",
        "Convective precipitation": "143_cp",
        "Surface net short-wave (solar) radiation": "15_msnswrf",
        "Total cloud cover": "45_tcc",
        "2 metre dewpoint temperature": "168_2d",
        "Maximum temperature at 2 metres since previous post-processing": "201_mx2t",
        "Minimum temperature at 2 metres since previous post-processing": "202_mn2t",
        "100 metre U wind component": "246_100u",
        "100 metre V wind component": "247_100v"
}
#        "Total precipitation": "logtp",

src = "extra_"
src = "extra12h_"
#src = "extra6h_"
#src = "extrarad_"
src = "extraprecip_"
src = "Extraprecip_"
src = "neoxtra"
if src == "Extra_":
    del mp["Large-scale precipitation"]
    del mp["Convective precipitation"]
    del mp["Surface net short-wave (solar) radiation"]
    del mp["Maximum temperature at 2 metres since previous post-processing"]
    del mp["Minimum temperature at 2 metres since previous post-processing"]

if src == "extra6h_" or src == "extra12h_":
    del mp["Large-scale precipitation"]
    del mp["Convective precipitation"]
    del mp["Sea surface temperature"]
    del mp["Total cloud cover"]
    del mp["Surface net short-wave (solar) radiation"]
    del mp["2 metre dewpoint temperature"]

if src == "extrarad_" or src == "extraprecip_":
    del mp["Sea surface temperature"]
    del mp["Total cloud cover"]
    del mp["2 metre dewpoint temperature"]
    del mp["Maximum temperature at 2 metres since previous post-processing"]
    del mp["Minimum temperature at 2 metres since previous post-processing"]

if src == "Extraprecip_":
    del mp["Sea surface temperature"]
    del mp["Total cloud cover"]
    del mp["2 metre dewpoint temperature"]

print("uhh", mp)
imp = {v: k for k, v in mp.items()}

mp = {"Sea surface temperature": "034_sstk",
        "Total cloud cover": "45_tcc",
        "2 metre dewpoint temperature": "168_2d",
        "100 metre U wind component": "246_100u",
        "100 metre V wind component": "247_100v"
}
#

refnan = np.load('/fast/proc/era5/extra/034_sstk/202101/1612134000.npz')['x']
whnan = np.where(np.isnan(refnan))

base = "/fast/ignored/mars_data/"
base = "/fast/ignored/mars_data_sfc/"
ls = sorted([x for x in os.listdir(base) if x.endswith(".grib") and src in x and not os.path.exists(base+x+".done")])
#ls = ["extraprecip_sfc_20220626.grib"]
bad = 0
for fn in ls:
    print("------------ file", fn, "------------------")
    donefn = base+fn+".done"
    f = pygrib.open(base+fn)
    i = 0
    l = len(f)
    sfc = np.zeros((720, 1440), dtype=np.float16)+np.nan
    saved = {}
    for grb in f:
        if grb.name in mp:
            vals = grb.values[:][:720]
            #print(grb.name, "uhhhh meanstd", np.mean(vals), np.std(vals))
            """
            if grb.name == "Sea surface temperature":
                print(grb.values)
                import pdb; pdb.set_trace()
                exit()
            """
            if grb.name == "Sea surface temperature":
                #vals[np.where(vals == 273.16015625)] = np.nan
                vals[whnan] = np.nan
            if "radiation" in grb.name:
                vals = vals/3600
            #print("hey", mp[grb.name] in norm, grb.P1, grb.P2)
            if grb.P1 in [5,11] and grb.P2 == 0:
                saved[grb.name] = (grb.analDate+timedelta(hours=grb.P1), vals)
                continue
            if grb.P2 in [5,11] and "temperature" in grb.name:
                continue
            if mp[grb.name] in norm:
                """
                print(grb)
                for x in grb.keys():
                    print(x, getattr(grb,x))
                exit()
                d = grb.validDate
                continue
                """
                #print(grb)
                if grb.P2 == 0 and grb.P1 != 0:
                    d = grb.analDate + timedelta(hours=grb.P1)
                elif grb.has_key("P2"):
                    d = grb.analDate + timedelta(hours=grb.P2)
                else:
                    assert False
                    d = grb.validDate
                #assert grb.P1 in [6,12]
                #print("grb", grb)
                #print("date", d)
                if grb.P1 in [6,12] and "temperature" not in grb.name:
                    S = saved[grb.name]
                    assert d-S[0] == timedelta(hours=1)
                    #print("subbing", vals.mean(), S[1].mean(), d, S[0])
                    vals = vals - S[1]

                if "precipitation" in grb.name:
                    vals = np.maximum(vals,0)
                    #print("min", np.min(vals), "mean", np.mean(vals), "pos", len(vals[np.where(vals>=0)].flatten())/(720*1440)*100)
                    vals = np.log(np.maximum(vals + 1e-7, 0))

                nn = norm[mp[grb.name]]
                normed = (vals - nn[0][0])/np.sqrt(nn[1][0])
                print("hey", grb.name, np.nanmean(normed, axis=(0,1)), np.nanstd(normed,axis=(0,1)), np.isnan(normed).sum(), d)
                normed = normed.astype(np.float16)

                bbase = "/fast/proc/hres/extra/%s/%04d%02d/" % (mp[grb.name], d.year, d.month)
                os.makedirs(bbase, exist_ok=True)
                out = bbase + "%d.tmp.npz" % (d-datetime(1970,1,1)).total_seconds()
                np.savez(out, x=normed)
                os.rename(out, out.replace(".tmp.npz", ".npz"))
            else:
                print("aaaa", grb.name)
                assert False
                pass#print("aaaa", grb.name)
        else:
            pass#print("missing", grb.name)
        continue
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
            outpath = "/fast/proc/hres/f000/%04d%02d/%d.npz" % (d.year, d.month, (d-datetime(1970,1,1)).total_seconds())
            era5 = "/fast/proc/era5/f000/%04d%02d/%d.npz" % (d.year, d.month, (d-datetime(1970,1,1)).total_seconds())
            if os.path.exists(outpath):
                try:
                    print("ref exists, comparing and not overwriting!!")
                    ref = np.load(outpath)
                    era5 = np.load(era5)
                    #plt.imshow(delta[:,:,0])
                    #plt.savefig("/fast/public_html/delta.png", bbox_inches='tight')
                    #print("rms pr", np.sqrt(np.mean(np.square(pr.astype(np.float32)-ref["pr"].astype(np.float32)), axis=(0,1))))
                    print("rms sfc", np.sqrt(np.mean(np.square(sfc.astype(np.float32)-ref["sfc"].astype(np.float32)), axis=(0,1))))
                    print("era5 new", np.sqrt(np.mean(np.square(sfc.astype(np.float32)-era5["sfc"].astype(np.float32)), axis=(0,1))))
                    print("era5 old", np.sqrt(np.mean(np.square(ref["sfc"].astype(np.float32)-era5["sfc"].astype(np.float32)), axis=(0,1))))
                    """
                    print("biasnew", np.mean(sfc.astype(np.float32), axis=(0,1)))
                    print("stdnew", np.std(sfc.astype(np.float32), axis=(0,1)))

                    print("biasold", np.mean(ref["sfc"].astype(np.float32), axis=(0,1)))
                    print("stdold", np.std(ref["sfc"].astype(np.float32), axis=(0,1)))
                    """
                except:
                    print("comparison failed")
                    traceback.print_exc()
            else:
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
    with open(donefn, "w") as f:
        pass

