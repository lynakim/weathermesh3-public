import os
import sys
import time

# run on halfmoon only
assert os.uname().nodename == 'halfmoon'
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('start_year', type=int)
parser.add_argument('end_year', type=int)
parser.add_argument('--hourly', type=int, default=6, help='Optional string argument')
args = parser.parse_args()


def gen_mods(hr):
    text = "\n".join([f"{nix}.npz" for nix in range(0,1728172800,3600*hr)])
    with open(f"{hr}hr.txt", "w") as f:
        f.write(text)

#gen_mods(args.hourly)
#exit()

def mkyr(year):
    return [f"{year}{month:02d}" for month in range(1,13)]

command = 'rclone copy --files-from=6hr.txt /fast/proc/era5/f000/{date}/ stratus:/jersey/era5/f000/{date}/ --progress --stats-one-line --multi-thread-streams=64 --transfers=64'

for year in range(args.start_year,args.end_year+1):
    dates = mkyr(year)
    for date in dates:
        print(f"====> {date}")
        os.system(command.format(date=date))
        time.sleep(0.2)
