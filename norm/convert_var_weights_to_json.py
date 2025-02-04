import json
import pickle
import numpy as np

variable_weights = pickle.load(open(f'/fast/consts/tc_variable_weights_28.pickle', 'rb'))

levels_medium = [10, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000]
out = {'levels': levels_medium}

keys = variable_weights.keys()

for k in keys:
    variable_weight = variable_weights[k]
    if isinstance(variable_weight, np.ndarray):
        variable_weight = variable_weight.tolist()
        variable_weight = [1/x for x in variable_weight]
        out[k] = variable_weight
    else:
        out[k] = 1 / (variable_weight * 0.2)

with open('norm/variable_weights.json', 'w') as f:
    json.dump(out, f, indent=4)