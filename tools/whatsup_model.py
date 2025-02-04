import os 
print(os.getcwd())

import sys
print(sys.executable)
print(sys.path)
sys.path.insert(0, '')
from evals.package_neo import *

model = get_rtbachelor() 

dataconf = NeoDataConfig(inputs=model.config.inputs,
                         ouputs=model.config.outputs,
                         requested_dates = get_dates((D(1900, 1, 1),D(2100, 1, 1))),
                         only_at_z = list(range(24))
                         )
data = NeoWeatherDataset(dataconf)
data.check_for_dates()
