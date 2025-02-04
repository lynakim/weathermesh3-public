import os
import json
import numpy as np

VAR = "167_2t"
VAR = "151_msl"
VAR = "166_10v"
base = "/fast/evaluation/hegelquad_333M/errors/"
fs = sorted([x for x in os.listdir(base) if "+24." in x if x >= "202208"])
#print(fs)
jj = [json.load(open(base+x)) for x in fs]
rms = np.sqrt(np.mean(np.square([j["rmse"][VAR] for j in jj])))
print("eyo rms", VAR, rms)
