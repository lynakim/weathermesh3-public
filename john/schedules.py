import train
from utils_lite import WeatherTrainerConfig
import matplotlib.pyplot as plt 
import numpy as np
config = WeatherTrainerConfig()
config.lr_sched.cosine_period = 60_000
config.lr_sched.warmup_end_step = 1000
config.lr_sched.lr = 0.3e-3

steps = np.arange(0, 60_000)
lr = np.array([train.WeatherTrainerBase.computeLR(step,config.lr_sched,n_step_since_restart=step) for step in steps])
max_dt = np.array([train.WeatherTrainerBase.computeMaxDT(step,config.lr_sched) for step in steps])
num_rand = np.array([train.WeatherTrainerBase.computeNumRandomSubset(step,config.lr_sched) for step in steps])

#three horizonal plots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(steps, lr)
axs[0].set_title('LR Schedule')
axs[0].grid()

axs[1].plot(steps, max_dt)
axs[1].set_title('Max DT Schedule')
axs[1].grid()

axs[2].plot(steps, num_rand)
axs[2].set_title('Num Random Subset Schedule')
axs[2].grid()

fig.tight_layout()
plt.savefig('ohp.png')

