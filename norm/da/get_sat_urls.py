from datetime import datetime, timedelta
import os

# aria2c --dir=/huge/proc/ncarsat -c -j16 -x16 -i sats1 --auto-file-renaming=false

d = datetime(2005,1,1)

OUTPUT_PATH = "/huge/proc/ncarsat"
srcs = ["1bamua", "gpsro", "atms", "mtiasi"]
#srcs = ["satwnd"]
srcs = ["adpupa"]
srcs = ["adpsfc"]

while d < datetime(2024,12,1):

    fmt = "https://data.rda.ucar.edu/d113001/ec.oper.%s/%04d%02d/ec.oper.%s.128_%s.regn1280sc.%04d%02d%02d%s.nc"
    fmt = "https://data.rda.ucar.edu/d735000/%s/%04d/%s.%04d%02d%02d.tar.gz"

    fmt2 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.%04d%02d%02d/%02d/gdas1.t%02dz.%s.tm00.bufr_d"

    fmt3 = "https://data.rda.ucar.edu/d351000/tarfiles/%04d/gdasupaobs.%04d%02d%02d.tar.gz"
    fmt4 = "https://data.rda.ucar.edu/d461000/tarfiles/%04d/gdassfcobs.%04d%02d%02d.tar.gz"
    #https://data.rda.ucar.edu/d735000/atms/2024/atms.20241121.tar.gz

    os.makedirs(f"{OUTPUT_PATH}/%04d%02d" % (d.year, d.month), exist_ok=True)

    for src in srcs:
        if d < datetime(2015, 10, 10) and src in ["gpsro", "atms", "iasi"]: continue
        url = fmt % (src, d.year, src, d.year, d.month, d.day)
        fn = url.split("/")[-1]
        if src in ["satwnd"]:
            url = fmt2 % (d.year, d.month, d.day, d.hour, d.hour, src)
            fn = fn.replace(".tar.gz", "%02d.bufr"%d.hour)
        if src in ["adpupa"]:
            url = fmt3 % (d.year, d.year, d.month, d.day)
        if src in ["adpsfc"]:
            url = fmt4 % (d.year, d.year, d.month, d.day)
        outf = "%04d%02d/%s"%(d.year, d.month, fn)

        if not os.path.exists(f"{OUTPUT_PATH}/"+outf):
            print(url+"\n\tout=%s"%outf)
         
    #https://data.rda.ucar.edu/d113001/ec.oper.an.pl/201601/ec.oper.an.pl.128_060_pv.regn1280sc.        2016010100.nc
    #fm2 = "https://data.rda.ucar.edu/d113001/ec.oper.an.sfc/201601/ec.oper.an.sfc.128_167_2t.regn1280sc.20160101.nc"
    #fm3 = "https://data.rda.ucar.edu/d113001/ec.oper.fc.sfc/201601/ec.oper.fc.sfc.128_142_lsp.regn1280sc.20160101.nc"

    d += timedelta(days=1)
    #d += timedelta(hours=6)

