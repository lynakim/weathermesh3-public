from utils import *
import json

norms = pickle.load(open(f'{CONSTS_PATH}/normalization.pickle', 'rb'))

out = {'levels': levels_full}

keys = ['129_z','130_t','131_u','132_v','133_q','rhlol']

for k in keys:
    out[k] = {'mean':[x for x in norms[k][0]],'std':[x**0.5 for x in norms[k][1]]}

with open('norm/norm.json','w') as f:
    json.dump(out,f,indent=4)
