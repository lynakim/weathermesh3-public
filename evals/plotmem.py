import json
import os 
from utils import *
import matplotlib.pyplot as plt


dpath = '/fast/windborne/deep/ignored/memuse/yo.json' 
data = []
with open(dpath,'r') as f:
    for i,l in enumerate(f.readlines()):
        data.append(json.loads(l))

def get(k):
    return [x[k] for x in data]

for k in ['alloc','max_alloc','reserv','max_reserv']:
    plt.plot(get(k),label=k)
plt.legend()
plt.grid()
plt.savefig(os.path.dirname(dpath)+'/memuse.png')


