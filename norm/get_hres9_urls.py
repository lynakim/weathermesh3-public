from datetime import datetime, timedelta
import os


d = datetime(2016,1,1)

OUTPUT_PATH = "/huge/proc/ncarhres"

while d < datetime(2024,11,1):

    pr = ["129_z", "130_t", "131_u", "132_v", "133_q"]
    sfc = ["165_10u", "166_10v", "167_2t", "151_msl", "034_sstk", "168_2d", "246_100u", "247_100v", "164_tcc"]
    fct = ["142_lsp", "143_cp", "176_ssr", "201_mx2t", "202_mn2t", "121_mx2t6", "122_mn2t6"]

    fmt = "https://data.rda.ucar.edu/d113001/ec.oper.%s/%04d%02d/ec.oper.%s.128_%s.regn1280sc.%04d%02d%02d%s.nc"

    os.makedirs(f"{OUTPUT_PATH}/%04d%02d" % (d.year, d.month), exist_ok=True)

    for a, bs in [("an.pl", pr), ("an.sfc", sfc), ("fc.sfc", fct)]:
        if a != "an.pl" and d.hour != 0: continue
        for b in bs:
            url = fmt % (a, d.year, d.month, a, b, d.year, d.month, d.day, ("%02d"%d.hour) if a == "an.pl" else "")
            if b in ['131_u', '132_v']:
                url = url.replace("1280sc", "1280uv")
            if b in ['246_100u', '247_100v']:
                url = url.replace("128_", "228_")

            fn = url.split("/")[-1]
            outf = "%04d%02d/%s"%(d.year, d.month, fn)

            if not os.path.exists(f"{OUTPUT_PATH}/"+outf):
                print(url+"\n\tout=%s"%outf)
         
    #https://data.rda.ucar.edu/d113001/ec.oper.an.pl/201601/ec.oper.an.pl.128_060_pv.regn1280sc.        2016010100.nc
    #fm2 = "https://data.rda.ucar.edu/d113001/ec.oper.an.sfc/201601/ec.oper.an.sfc.128_167_2t.regn1280sc.20160101.nc"
    #fm3 = "https://data.rda.ucar.edu/d113001/ec.oper.fc.sfc/201601/ec.oper.fc.sfc.128_142_lsp.regn1280sc.20160101.nc"

    d += timedelta(hours=6)

