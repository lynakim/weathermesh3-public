import os
import subprocess

ma = [("barceloneta", 6), ("miramar", 6), ("bimini", 6)]
ma = [("halfmoon", 4), ("stinson", 4), ("singing", 5)]
ss = sum(x[1] for x in ma)
cum = 0
GOOD_GPUS = {"singing": "0,1,2,4,5"}
for a, b in ma:
    gpus = list(range(b))
    if a in GOOD_GPUS:
        gpus = [int(x) for x in GOOD_GPUS[a].split(",")]
    for i in range(b):
        lcmd = "cd /fast/wbjoan10/deep && source /home/windborne/.bashrc && CUDA_VISIBLE_DEVICES=%d python3 -u runs/retro.py %d %d" % (i, cum, ss)
        cmd = ['ssh', a, f'bash -l -c "{lcmd}"']
        cum += 1
        print("gonna do", cmd)
        subprocess.Popen(cmd)
