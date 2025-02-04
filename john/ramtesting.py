
from evals.package_neo import *
from torch.utils.data import DataLoader

model = get_bachelor(None) 
timesteps = [1,2,6,7,11,12,13,18,19,24,25,30,31,48,132,133]

dataconf = NeoDataConfig(inputs=model.config.inputs,
                         ouputs=model.config.outputs,
                         timesteps = timesteps,
                         no_neoloader = True,
                         random_timestep_subset = 5,
                         requested_dates = get_dates((D(1900, 1, 1),D(2100, 1, 1))),
                         only_at_z = list(range(24))
                         )
data = NeoWeatherDataset(dataconf)
data.check_for_dates()


dataloader = DataLoader(data,num_workers=12,shuffle=True)

def print_tensor_shapes(data):
    if isinstance(data, torch.Tensor):
        return f"Tensor{list(data.shape)}"
    elif isinstance(data, list):
        return f"[{', '.join(print_tensor_shapes(item) for item in data)}]"
    else:
        return f"Unknown({type(data).__name__})"


t1 = time.time()
samples_per_sec = 0
for sample in dataloader:
    
    t2 = time.time()
    samples_per_sec = samples_per_sec*0.8 + 0.2*((t2-t1))
    t1 = t2
    print(f"got sample, {samples_per_sec:.2f}secs per sample")
