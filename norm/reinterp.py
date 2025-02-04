import sys
sys.path.append('..')
from utils import *

old = pickle.load(open(f'{CONSTS_PATH}/variable_weights_14.pickle', 'rb'))
pprint(old)
neo = {}
for d in old:
    if type(old[d]) not in [np.float32, np.float64, float]:
        neo[d] = np.interp(levels_medium, levels_small, old[d])
    else:
        neo[d] = old[d]
pprint(neo)
pickle.dump(neo, open(f'{CONSTS_PATH}/variable_weights_28.pickle', 'wb'))
