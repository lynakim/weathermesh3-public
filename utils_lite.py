
# this file exists becasue I'm sick of always importing torch every time 
# I need utils for ever script so that fast stuff is all going here
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
import os
import time
import numpy as np
import copy

D = lambda *x: datetime(*x, tzinfo=timezone.utc)

def RED(text): return f"\033[91m{text}\033[0m"
def GREEN(text): return f"\033[92m{text}\033[0m"
def YELLOW(text): return f"\033[93m{text}\033[0m"
def BLUE(text): return f"\033[94m{text}\033[0m"
def MAGENTA(text): return f"\033[95m{text}\033[0m"
def CYAN(text): return f"\033[96m{text}\033[0m"
def WHITE(text): return f"\033[97m{text}\033[0m"
def ORANGE(text): return f"\033[38;5;214m{text}\033[0m"

ASNI_COLORS = {
    'RED' : '\033[91m',
    'GREEN' : '\033[92m',
    'YELLOW' : '\033[93m',
    'BLUE' : '\033[94m',
    'MAGENTA' : '\033[95m',
    'CYAN' : '\033[96m',
    'WHITE' : '\033[97m',
    'END' : '\033[0m',
    'ORANGE': '\033[38;5;214m',
}

default_hres_config = None
try:
    with open("hres/hres_utils.py") as f:
        t = f.read().replace("default_config", "default_hres_config")
        exec(t)
except:
    pass

class LRScheduleConfig:
    def __init__(self, conf_to_copy=None, **kwargs):
        self.lr = 2e-4
        self.warmup_end_step = 1000
        self.restart_warmup_end_step = 100
        self.cosine_en = True
        self.cosine_period = 45_000
        self.cosine_bottom = 5e-8
        self.step_offset = 0
        self.div_factor = 4
        self.schedule_dts = False
        self.max_dt_min = 24
        self.max_dt_max = 144
        self.steps_till_max_dt_max = 45_000
        self.num_random_subset_min = 2
        self.num_random_subset_max = 8
        self.steps_till_num_random_subset_max = 45_000
        self.schedule_dts_warmup_end_step = 1000

        if conf_to_copy is not None:
            self.__dict__.update(copy.deepcopy(conf_to_copy.__dict__))
        for k, v in kwargs.items():
            assert hasattr(self, k), f"Unknown LRSchedulerConfig option: {k}"
            setattr(self, k, v)

    def computeLR(self,step, n_step_since_restart=None):

        is_restart = n_step_since_restart != step and n_step_since_restart is not None
        cycle = self.warmup_end_step + self.cosine_period
        step = max(step + self.step_offset,0)
        n = step // cycle
        step_modc = step % cycle
        lr = np.interp(step+1, [0, self.warmup_end_step], [0, self.lr])
        if step > self.warmup_end_step:
            if self.cosine_en:
                lr = np.interp(step_modc+1, [0, self.warmup_end_step], [0, self.lr / (self.div_factor**n)]) # should use og warmup step to get og lr curve
                cstep = step_modc - self.warmup_end_step
                lr = lr * (np.cos(cstep/self.cosine_period *np.pi)+1)/2
                if self.cosine_bottom is not None:
                    if n > 0:
                        lr = self.cosine_bottom
                    else:
                        lr = max(lr, self.cosine_bottom)
            else: 
                lr = self.lr
        if is_restart and n_step_since_restart < self.restart_warmup_end_step:
            return lr * n_step_since_restart / self.restart_warmup_end_step
        return lr

    def computeMaxDT(self,step,proc_dt = 6, slow=False):
        if not self.schedule_dts:
            return 0
        if type(proc_dt) == list:
            assert len(proc_dt) == 1, "proc_dt must be a single integer for random timesteps"
            proc_dt = proc_dt[0]
        if step < self.schedule_dts_warmup_end_step:
            if slow:
                return 2*proc_dt
            else:
                return proc_dt
        self.steps_till_max_dt_max = min(self.steps_till_max_dt_max, self.cosine_period)
        max_dt = self.max_dt_min + (self.max_dt_max - self.max_dt_min)/self.steps_till_max_dt_max**2 * min(step,self.steps_till_max_dt_max)**2
        max_dt = max_dt // proc_dt * proc_dt
        return int(max_dt)
    
    def computeNumRandomSubset(self,step, slow=False):
        if not self.schedule_dts:
            return 1
        if step < self.schedule_dts_warmup_end_step:
            if slow:
                return 3
            else:
                return 1
        self.steps_till_num_random_subset_max = min(self.steps_till_num_random_subset_max, self.cosine_period)
        num_random_subset = self.num_random_subset_min + (self.num_random_subset_max - self.num_random_subset_min)/self.steps_till_num_random_subset_max * min(step,self.steps_till_num_random_subset_max)
        return int(num_random_subset)
    
    def make_plots(self, step_range=None, save_path="ignored/plots/lr_sched.png"):
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        if step_range is None:
            step_range = self.cosine_period
        _, axs = plt.subplots(3,1,figsize=(6,10))
        axs[0].plot(np.arange(0,step_range,100),np.vectorize(self.computeLR)(np.arange(0,step_range,100))); axs[0].grid(); axs[0].set_title("LR") ; axs[0].set_xlabel("Step")
        axs[1].plot(np.arange(0,step_range,100),np.vectorize(self.computeMaxDT)(np.arange(0,step_range,100))); axs[1].grid(); axs[1].set_title("Max DT") ; axs[1].set_xlabel("Step")
        axs[2].plot(np.arange(0,step_range,100),np.vectorize(self.computeNumRandomSubset)(np.arange(0,step_range,100))); axs[2].grid(); axs[2].set_title("Num Random Subset") ; axs[2].set_xlabel("Step")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

class WeatherTrainerConfig():
    def __init__(self,conf_to_copy=None,**kwargs):
        self.yolo = False
        self.activity = ''
        self.nope = False
        self.no_logging = False
        self.resume = False
        self.resume_select = ''
        self.new_resume_folder = False
        self.quit = False
        self.name = ''
        self.prefix = ''
        self.gpus = '0'
        self.dimprint = False
        self.reset_optimizer = False
        self.reset_steps_on_resume = False
        self.batch_size = 1
        self.slow_start = False # uses more steps for the first 1k steps to avoid big jumps
        self.num_workers = 2
        self.prefetch_factor = 2
        self.pin_memory = True
        self.diff_loss = 0.0
        self.HALF = True # Half precision
        self.TIME = False 
        self.log_every = 25 # How often to log to tensorboard
        self.log_step_every = 25
        self.save_every = 100 # How often to save model
        self.save_optimizer = True
        self.optim = 'shampoo'
        self.actually_scale_gradients = True
        self.initial_gradscale = 65536.0
        self.latent_l2 = 0
        self.clamp = 13
        self.clamp_output = None
        self.steamroll_over_mismatched_dims = False
        self.adam = SimpleNamespace()
        self.adam.betas= (0.9, 0.99)
        self.adam.weight_decay= 0.001
        self.dt_loss_weights_override = {}
        self.shampoo = SimpleNamespace()
        self.shampoo.version = 'old'
        self.shampoo.dim = 8192
        self.shampoo.num_trainers = -1 # defaults to # gpus
        self.dt_loss_beta = 0.995
        self.lr_sched = LRScheduleConfig()
        self.drop_path = False
        self.drop_sched = SimpleNamespace()
        self.drop_sched.drop_max = 0.2
        self.drop_sched.drop_min = 0.0
        self.drop_sched.iter_offset = -816_300
        self.drop_sched.ramp_length = 200_000
        self.N_epochs = 1_000_000_000
        self.cpu_only = False
        self.val_date_range = [D(2018, 1, 1), D(2018, 12, 30)]
        self.only_at_z = None 
        self.seed = 0
        self.shuffle = True
        self.coupled = SimpleNamespace()
        self.coupled.hell = False
        self.coupled.weight = 1.0
        self.coupled.B = 512
        self.coupled.freeze = True
        self.coupled.config = default_hres_config
        self.ignore_train_safegaurd = False
        self.rerun_sample = 1 
        self.console_log_path = None
        self.skip_audit = False
        self.disregard_buffer_checksum = False
        self.use_tf32 = False
        self.on_cloud = False
        self.timeout = timedelta(minutes=10)
        self.profile = False
        self.print_ram_usage = False
        self.save_imgs_every = 1_000
        self.compute_Bcrit_every = np.nan
        self.diffusion = False
        self.bboxes = None
        self.strict_load = True
        self.use_point_dataset = False
        self.point_batch_size = 16
        self.find_unused_params = False
        self.do_da_compare = False

        if conf_to_copy is not None:
            self.__dict__.update(copy.deepcopy(conf_to_copy.__dict__))
        for k,v in kwargs.items():
            assert k != 'val_dates', "Use val_date_range instead of val_dates. Only provide a list of two dates."
            assert hasattr(self,k), f"Unknown config option: {k}"
            setattr(self,k,v)

def am_i_torchrun():
    return 'TORCHELASTIC_ERROR_FILE' in os.environ.keys()

