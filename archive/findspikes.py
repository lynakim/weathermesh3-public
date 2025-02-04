import os
import re

runs = os.listdir('ignored/runs')
for r in runs:
    try:
        f = open('ignored/runs/'+r+'/log.txt').read()
    except:
        print("no log for", r)
        continue
    losses = [(int(x[1]), float(x[2]), x[0]) for x in re.findall('(.*?) (\d+) . Loss:(.*)', f)]
    last = None
    for a, b, c in losses:
        if last is not None:
            if b/last > 10:
                print("Spike!!", a, b, c, r)
        last = b
