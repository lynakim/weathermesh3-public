from datetime import datetime, timedelta

years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
#years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025] # 60
#years = [2023, 2024]

for y in years:
    d = datetime(y, 1, 1)
    while d.year == y:
        #print("python3 norm/da/procwindborne.py %04d-%02d-%02d" % (d.year, d.month, d.day))
        #print("python3 norm/da/procsatwnd.py adpupa satwnd %04d%02d%02d" % (d.year, d.month, d.day))
        print("python3 norm/da/procradiosonde.py adpupa adpupa %04d%02d%02d" % (d.year, d.month, d.day))
        d += timedelta(days=1)

# then sth like "parallel --ungroup --jobs 32 < list_of_cmds"
# python3 norm/da/makecmds.py | shuf | parallel --ungroup --jobs 16
# nice -n 10 python3 norm/da/makecmds.py | shuf | parallel --ungroup --jobs 32