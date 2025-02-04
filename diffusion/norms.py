import re
import pickle
from tqdm import tqdm
import numpy as np
import os
from collections import defaultdict
import sys
from pprint import pprint
import json

levels_full = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]

with open("/fast/consts/normalization.pickle", "rb") as f:
    norm = pickle.load(f)

def get_norms(model, DH, savename=None):
    if savename is None:
        savename = model

    base = '/huge/deep/evaluation/%s/errors'%model
    ls = sorted([x for x in os.listdir(base) if re.search('\+%d\.vs'%DH, x) and x.startswith('2020')])

    Var = defaultdict(list)
    Mean = defaultdict(list)
    for f in tqdm(ls):
        with open(base+"/"+f) as f:
            j = json.loads(f.read())
        for k in j["rmse"]:
            mean = j["bias"][k]
            var = j["rmse"][k]**2 - mean**2
            if '133_q' in k:
                mean /= 1e6
                var /= 1e12
            Mean[k].append(mean)
            Var[k].append(var)
    
    for k in list(Mean.keys()):
        Var[k] = np.mean(Var[k]) + np.var(Mean[k])
        Mean[k] = np.mean(Mean[k])

    for a in ["129_z", "130_t", "131_u", "132_v", "133_q","zeropad"]:
        if a in Mean:
            del Mean[a]


    scale = {}

    for k in list(Mean.keys()):
        std = np.sqrt(Var[k])
        mean = Mean[k]
        fuckit = np.sqrt(std**2 + mean**2) # fuck having to deal with offsets
        try: nn = np.sqrt(norm[k][1])[0]
        except: nn = np.sqrt(norm['_'.join(k.split("_")[:-1])][1][levels_full.index(int(k.split("_")[-1]))])
        scale[k] = fuckit/nn
        print(k, scale[k])
    with open("/fast/consts/diffusion_scaling/stds_%s_%d.pickle" % (savename, DH), "wb") as f:
        pickle.dump(scale, f)

    return scale


#print(get_norms("ultralatentbachelor_168M", 24,savename="serp3bachelor"))
print(get_norms("joansucks", 24))



