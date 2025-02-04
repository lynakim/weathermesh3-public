#from utils import *
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import torch.utils.checkpoint as checkpoint
import torch.distributed.checkpoint as dist_checkpoint
import torch
import os
import meshes
import torch
import sys
from data import *
from train_fn import *
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader    
import torch.profiler
import inspect
#from model_latlon import *
from model_latlon_3d import *
from eval import *
from utils import print
import importlib
import GPUtil
import socket
import torch.multiprocessing as mp
from hashlib import sha256
import threading
import psutil
from functools import partial
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
#from hres.data import HresDataset      # yo if you add tihs back in, have an if to only inpor if coupled learning
#from hres.train import Trainer as HresTrainer
from collections import deque, Counter
from model_latlon.top import *
from train.diffusion_methods import DiffusionMethods
from train.train_methods import *
from model_latlon.point_dec import POINT_DECODERS
from distributed_shampoo.distributed_shampoo import DistributedShampoo


#torch.manual_seed(0)

def log(*args):
    frame = inspect.currentframe().f_back
    string = inspect.getframeinfo(frame).code_context[0].strip()
    arg_names = string[string.find('(') + 1:-1].split(',')

    for arg, value in zip(arg_names, args):
        print(f"{arg.strip()}: {value}", end=" ")
    print("")

# Looking for WeatherTrainerConfig? Sorry, the princess is in another castle. Try utils_lite.py


# annoying code below for getting PIDs of workers for the torch dataloader
def worker_init_fn(worker_id,pid_dict):
    pid = os.getpid()
    pid_dict[worker_id] = pid
    print(f"Worker {worker_id} has pid {pid}")


# Define collate_fn for data loading scheme (defined in data.py)
def collate_fn(batch):
    # TODO: make this work with batch size > 1 for StationData
    # Dataloader currently assumes we always have a batch size of 1 (i.e. only one tensor per instance)
    assert len(batch) == 1, "Batch size must be 1"
    batch = batch[0]
    
    for input_index, input_data in enumerate(batch.inputs):
        mesh_ids, tensors = input_data
        collated_tensors = [tensor.unsqueeze(0) for tensor in tensors]
        batch.inputs[input_index] = [mesh_ids, collated_tensors]
        
    for output_index, output_data in enumerate(batch.outputs):
        mesh_ids, tensors = output_data
        collated_tensors = []
        for tensor in tensors:
            if type(tensor) == torch.Tensor:
                tensor = tensor.unsqueeze(0)
            collated_tensors.append(tensor)
        batch.outputs[output_index] = [mesh_ids, collated_tensors]
        
    return batch

class WeatherTrainerBase():

    def LRsched(self,step):
        
        c = self.conf.lr_sched
        lr = c.computeLR(step, self.n_step_since_restart)
        self.current_lr = lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr

class WeatherTrainer(WeatherTrainerBase,SourceCodeLogger,DiffusionMethods):

    @TIMEIT()
    def __init__(self,data,model=None,conf=None):
        self.step_offset = 0
        self.last_diff = None
        self.model = model
        self.data = data
        if conf is None: self.conf = WeatherTrainerConfig()
        else: self.conf = conf
        self.DDP = am_i_torchrun()
        if self.DDP:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        gpus = ["cuda:"+str(i) for r in self.conf.gpus.split(',') for i in range(int(r.split('-')[0]), int(r.split('-')[-1]) + 1)]
        if self.DDP:
            if len(gpus) == 1:
                self.gpus = ["cuda:"+str(self.local_rank)]
            else:
                self.gpus = [gpus[self.local_rank]]
        else:
            self.gpus = gpus
        self.gpus = self.gpus[:torch.cuda.device_count()]
        if len(self.gpus) == 0:
            self.gpus = ["cpu"]
        self.num_gpus = len(self.gpus)
        if self.model is None: self.model = NoOp()

        self.state = SimpleNamespace()
        self.state.n_iter = 0
        self.state.n_step = 0
        self.state.num_restarts = 0
        self.n_step_since_restart = 0
        self.writer = NoOp()
        self.weight = None
        self.do_audit = False
        self.audit_stats = SimpleNamespace()
        self.audit_stats.checked_inputs = False
        self.dt_loss_weights = {}

        # Make sure we don't set timesteps and schedule_dts
        if self.data.config.timesteps != [24]: assert self.conf.lr_sched.schedule_dts is False, "Cannot set both timesteps and schedule_dts"

        if self.conf.nope:
            print(RED("❌❌❌ Nope=true, log folder will be in /tmp/!"))
        else:
            print(GREEN("✅✅✅ Nope=false, log folder will be on tensorboard!"))

        print(MAGENTA(f"Setting up WeatherTrainer. self.model.config.checkpoint_type is 👉{self.model.config.checkpoint_type}👈 Is this what you want?"))            

        if not self.DDP and self.conf.optim == "shampoo":
            self.conf.optim = "adam"
            print("Shampoo not supported without DDP, switching to adam")
        #torch.autograd.set_detect_anomaly(True)

    def run(self):
        print(YELLOW(f"Run Starting {time.time()-PROC_START_TIME}s since process start"))
        self.setup()
        while 1:
            self.train()

    ############ SETUP ############

    def setup(self):
        self.setup_data()
        self.setup_logging()
        self.setup_training()

        # TODO: remove the below if possible
        self.checkpoint = None
        del self.checkpoint
        import gc
        gc.collect()
    
    @TIMEIT()
    def setup_data(self):
        c = self.conf
        self.rank = 0
        self.world_size = 1
        self.primary_compute = torch.device(self.gpus[0]) 
        if "CUDA_LAUNCH_BLOCKING" in os.environ: print(f'CUDA_LAUNCH_BLOCKING is set to {os.environ["CUDA_LAUNCH_BLOCKING"]}', only_rank_0=True)
        if "NCCL_DEBUG" in os.environ: print(f'NCCL_DEBUG is set to {os.environ["NCCL_DEBUG"]}', only_rank_0=True)
        torch.backends.cuda.matmul.allow_tf32 = c.use_tf32
        if self.DDP:
            dist.init_process_group('nccl', timeout=c.timeout)
            assert len(self.gpus) == 1, "DDP only works with one GPU per process"
            self.rank = dist.get_rank()
            if self.rank == 0:
                print('use_tf32 is '+ str(c.use_tf32))
                print(f'DDP timeout due to nccl error is set to {c.timeout} seconds')
            self.world_size = dist.get_world_size()
            self.primary_compute = torch.device(self.gpus[0])
            torch.cuda.set_device(self.primary_compute)
            self.num_gpus = self.world_size
            print('use_tf32 is '+ str(c.use_tf32), only_rank_0=True)        
            import natten
            if hasattr(natten, 'WB_MODDED'):
                print(RED(f"Using {'NATTEN' if hasattr(natten, 'WB_MODDED') else 'FNA'}"), only_rank_0=True)
            print("Starting DDP with rank",self.rank, "world size", self.world_size)
        elif self.conf.cpu_only:
            self.primary_compute = torch.device('cpu')
 
        if '_' in self.conf.name: print("WARNING: underscore in name messes with things, replacing with dash", only_rank_0=True)
        self.conf.name = self.conf.name.replace('_','-')

        self.log_dir = (self.conf.nope*"/tmp/")+f"{RUNS_PATH}/runs{self.conf.prefix}/run_{self.conf.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"  
        if self.DDP:
            brdcast = [self.log_dir if self.rank == 0 else None]
            dist.broadcast_object_list(brdcast, src=0)
            self.log_dir = brdcast[0]

        self.setup_resume()

        self.data.config.log_dir = self.log_dir
        #print("hey seed is", torch.seed())
  
        self.print_mem('before model load')
        time.sleep(0.1*self.rank) # get in line GPUs
        self.model = self.model.to(self.primary_compute)
        print(f'Loaded model to {self.primary_compute}')
        self.print_mem('after model load')

        self.generator = torch.Generator()
        self.val_generator = torch.Generator()
        self.data.config.wold_size = self.world_size
        self.data.config.rank = self.rank
        self.data_loader = [0] # has to default to that can be indexed 

        if self.DDP:
            # there is no speedup in turning off find unused parameters, but the warning is annoying so we turn it off
            # https://chat.windbornesystems.com/#narrow/stream/201-tech-dlnwp/topic/misc/near/6031252
            warnings.filterwarnings("ignore","find_unused_parameters=True was specified in DDP constructor")
            self.active_model = DistributedDataParallel(
                self.model,device_ids=[self.primary_compute],
                find_unused_parameters=self.conf.find_unused_params)
            self.model = self.active_model.module
        else:
            assert self.num_gpus == 1, "Only one GPU supported without DDP"
            self.active_model = self.model 

        self.N_training_samples = len(self.data_loader)



    @TIMEIT()
    def setup_resume(self):
        self.last_pid = None
        logp = os.path.join(*os.path.split(self.log_dir)[:-1])
        self.checkpoint = None
        if self.conf.resume is False: return 
        if self.conf.resume is None: self.conf.resume = ''

        if '/' in self.conf.resume:
            assert os.path.exists(self.conf.resume), f"Resume path does not exist: {self.conf.resume}"
            assert os.path.isfile(self.conf.resume), f"Resume path is not a file: {self.conf.resume}"
            lpath = self.conf.resume
            resume_dir = os.path.dirname(lpath)
        else: 
            runs = [x for x in os.listdir(logp) if self.conf.resume in x]
            assert len(runs) != 0,f"No runs in {logp} found matching resume arg: {self.conf.resume}"
            dates = [x.split('_')[-1] for x in runs]
            recent = runs[max(enumerate(dates), key=lambda x: x[1])[0]]
            resume_dir = os.path.join(logp,recent)
            saves = [x for x in os.listdir(resume_dir) if x.endswith('.pt')]
            assert len(saves) != 0,f"No saves found in resume dir: {resume_dir}"
            if self.conf.resume_select == 'loss':
                losses = [float(x.split('loss')[-1].split('.pt')[0]) for x in saves]
                furthest_save = saves[min(enumerate(losses), key=lambda x: x[1])[0]]
            elif self.conf.resume_select.startswith('iter') or self.conf.resume_select == '':
                # to remain back compatible with filenames without _step
                iters = [int(x.split('_iter')[-1].split('_step')[0]) if '_step' in x else int(x.split('_iter')[-1].split('_loss')[0]) for x in saves]
                if ':' in self.conf.resume_select:
                    tgt = int(self.conf.resume_select.split(':')[1])
                    furthest_save = saves[np.argmin(np.abs(np.array(iters) - tgt))]
                else:
                    furthest_save = saves[max(enumerate(iters), key=lambda x:x[1])[0]]
            else:
                assert False, "Custom resume select not implemented yet"  
            lpath = os.path.join(resume_dir,furthest_save)
        try:
            self.loadCheckpoint(lpath)
        except torch.cuda.OutOfMemoryError as e:
            print("Out of memory loading checkpoint")
            raise e
        except Exception as e:
            print("Failed to load from checkpoint. Could try loading the old src, but I'm scared")
            raise e

        if not self.conf.new_resume_folder:
            self.log_dir = resume_dir

    @TIMEIT()
    def loadCheckpoint(self,lpath):
        print("Loading checkpoint from:", lpath)
        self.print_mem('before checkpoint load')
        #self.checkpoint = torch.load(lpath,map_location=self.primary_compute)
        self.checkpoint = torch.load(lpath,map_location='cpu')
        self.print_mem('after checkpoint load')

        if 'state' in self.checkpoint:# and not self.conf.reset_optimizer:
            update_namespace(self.state, self.checkpoint['state'])
        if self.conf.reset_steps_on_resume:
            self.state.n_step = 0
            self.state.n_iter = 0
        if 'buffer_checksum' in self.checkpoint and not self.conf.disregard_buffer_checksum:
            cp = self.checkpoint['model_state_dict']
            """
            nb = self.model.named_buffers()
            for name, param in self.model.state_dict().items():
                if name in nb: continue
                print(name, param.shape, name in cp)
                #assert name in cp
                if name in cp:
                    print(param.shape, cp[name].shape)
                    assert param.shape == cp[name].shape
            """

            assert self.get_buffers_checksum(_print=True) == self.checkpoint['buffer_checksum'], "Model buffer checksum does not match!"
            
        self.print_mem('after checkpoint load 2')
        self.state.num_restarts += 1
        #if self.DDP: assert False, "Need to look at this"
        state_dict = model_state_dict_name_mapping(self.checkpoint['model_state_dict'], self.model)
        
        
        bad = 207.273, np.sqrt(5108.7202)
        good = 291.638, np.sqrt(105.98)
        if str(self.conf.steamroll_over_mismatched_dims).startswith("Forgive me father for I have sinned"):
            new_state_dict = self.model.state_dict()
            """
            W = 6
            ofs = torch.sum(state_dict["encoder.conv_sfc.weight"][:,W] * (bad[0] - good[0])/bad[1], axis=(1,2))
            #if self.rank == 0: print("prev", state_dict["encoder.conv_sfc.bias"][:, 6], "ofs", ofs)
            state_dict["encoder.conv_sfc.bias"] -= ofs
            state_dict["encoder.conv_sfc.weight"][:, W] *= good[1]/bad[1]
            state_dict["singledec.deconv_sfc.weight"][:, W] *= bad[1]/good[1]
            state_dict["singledec.deconv_sfc.bias"][W] -= (good[0] - bad[0])/good[1]
            """
            for name, param in state_dict.items():
                if self.rank == 0: print(name, param.shape)
                if param.shape != new_state_dict[name].shape:
                    if name == "encoder.conv_sfc.weight":
                        cpy = new_state_dict[name].clone()
                        cpy[:, :4+12] = state_dict[name][:, :4+12]
                        cpy[:, 4+12+2:] = state_dict[name][:, 4+12:]
                        state_dict[name] = cpy
                    else:
                        if self.rank == 0:
                            print("!!! mismatch on", name, param.shape, new_state_dict[name].shape)
                        sl = tuple([slice(0, x) for x in param.shape])
                        if self.rank == 0: print("slicing", sl)
                        cpy = new_state_dict[name].clone()
                        cpy[sl] = state_dict[name][sl]
                        state_dict[name] = cpy
            state_dict["singledec.deconv_sfc.weight"][:, 16:18] = state_dict["singledec.deconv_sfc.weight"][:, 0:2]
            state_dict["singledec.deconv_sfc.bias"][16:18] = state_dict["singledec.deconv_sfc.bias"][0:2]
        
        if not self.conf.strict_load: print(ORANGE("😱😱😱 WARNING: strict_load is false, make sure you mean to do this 😱😱😱"))
        self.model.load_state_dict(state_dict,strict=self.conf.strict_load) 
        self.print_mem('after checkpoint load 3')

    @TIMEIT()
    def setup_logging(self):
        print("logging to", self.log_dir, only_rank_0=True)
        if not self.conf.no_logging:
            losslogs = os.path.join(self.log_dir, 'losslogs/')
            os.makedirs(losslogs, exist_ok=True)
            self.lossfile = open(losslogs + "/%s_%d.log" % (HOSTNAME, self.rank), "a")
            datelogs = os.path.join(self.log_dir, 'dataloader/dates/')
            os.makedirs(datelogs, exist_ok=True)
            self.datesfilepath = os.path.join(datelogs,f"{HOSTNAME}{self.rank}.log")

        if self.rank != 0: return
        os.makedirs(self.log_dir, exist_ok=True)
        if self.conf.console_log_path is not None:
            print("Symlinking console log to", self.conf.console_log_path)
            os.run(f"ln -s {self.log_dir}/consolelog{self.state.num_restarts}.{os.path.split(self.conf.console_log_path)[-2]} {self.conf.console_log_path}")

        if self.conf.no_logging:
            return
        self.keepsave_path = os.path.join(self.log_dir,'keepsave.txt')
        open(self.keepsave_path,'a').close()
        def add_code(self,name,code):
            self.add_text(name, f'```\n{code}\n```')
        SummaryWriter.add_code = add_code

        self.writer = SummaryWriter(log_dir=self.log_dir)
        with open(sys.argv[0], 'r') as f:
            run_code = f.read()
        self.writer.add_code('Run File', run_code)
        self.writer.add_text(f'Cmd String {self.state.num_restarts}',' '.join(sys.argv))
        self.writer.add_text(f'Hostname {self.state.num_restarts}',socket.gethostname())
        self.writer.add_text('model',str(self.model))
        #self.writer.add_text('timesteps',str(self.conf.timesteps))
        i = 0
        while 1:
            src_path = os.path.join(self.log_dir,f'src{i}')
            if not os.path.exists(src_path):
                os.makedirs(src_path, exist_ok=True)
                break
            i+=1
        os.system(f'cp ./*.py {src_path}/')
        self.writer.add_text(f'Source Path {self.state.num_restarts}',src_path)

        """
        dummy_input = torch.randn((1, 360, 720, self.model.D)).to(gpu)
        y = self.model(dummy_input)
        self.writer.add_graph(self.model, dummy_input)
        """
        self.writer.add_code('Model Source', self.model.get_source()) 
        self.writer.add_code('Trainer Source', self.get_source()) 


    @TIMEIT()
    def setup_training(self):    
        #net.to('cuda:0')
        #net = self.model.to(gpu)
        params = sum(p.numel() for p in self.active_model.parameters())
        print(ORANGE("Num params: %.2fM" % (params/1e6)), only_rank_0=True)
        paramst = sum(p.numel() for p in self.active_model.parameters() if p.requires_grad)
        print(ORANGE("Num trainable params: %.2fM" % (paramst/1e6)), only_rank_0=True)

        c = self.conf

        self.scaler = torch.cuda.amp.GradScaler(init_scale=self.conf.initial_gradscale)
        if not self.conf.actually_scale_gradients:
            self.scaler.set_growth_interval(2_000_000_000)
            self.scaler.set_growth_factor(1.25)
        else:
            self.scaler.set_growth_interval(500)

        self.print_mem('before optimizer')

        lparams = [x for x in self.active_model.parameters() if x.requires_grad]

        match self.conf.optim:
            case 'adam':
                self.optimizer = torch.optim.AdamW(lparams, **vars(self.conf.adam))
            case 'shampoo':
                if self.conf.shampoo.version == 'old': 
                    from distributed_shampoo.utils.shampoo_utils import GraftingType
                    self.optimizer = DistributedShampoo(
                    lparams,
                    lr=1e-4,
                    betas=self.conf.adam.betas,
                    epsilon=1e-12,
                    weight_decay=1e-6,
                    max_preconditioner_dim=self.conf.shampoo.dim,
                    precondition_frequency=1, # you probably want to start with this at like, 75!
                    start_preconditioning_step=1,
                    use_decoupled_weight_decay=True,
                    grafting_type=GraftingType.ADAM,
                    grafting_epsilon=1e-08,
                    grafting_beta2=0.999,
                    num_trainers_per_group=self.conf.shampoo.num_trainers,
                    )
                else: #new shampoo
                    #assert False
                    from distributed_shampoo.shampoo_types import AdamGraftingConfig, DDPShampooConfig, CommunicationDType
                    self.optimizer = DistributedShampoo(
                    lparams,
                    lr=1e-4,
                    betas=self.conf.adam.betas,
                    epsilon=1e-12,
                    weight_decay=1e-6,
                    max_preconditioner_dim=self.conf.shampoo.dim,
                    precondition_frequency=1, # you probably want to start with this at like, 75!
                    start_preconditioning_step=1,
                    use_decoupled_weight_decay=True,
                    grafting_config=AdamGraftingConfig(beta2=0.999, epsilon=1e-08),
                    distributed_config=DDPShampooConfig(communication_dtype=CommunicationDType.FP16,
                                                        communicate_params=False,
                                                        num_trainers_per_group=self.conf.shampoo.num_trainers,),
                    )

            case _:
                assert False, f'self.conf.optim: {self.conf.optim}. Not implemented'
        
        self.print_mem('after optimizer')
        if self.checkpoint is not None and not self.conf.reset_optimizer and 'optimizer_state_dict' in self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict']) 

        self.k2p = list(self.active_model.named_parameters())
        ns = [x[0] for x in self.k2p]
        assert len(ns) == len(set(ns))

        
        if not self.conf.reset_optimizer and self.checkpoint is not None:
            sd = {}
            if self.conf.optim == "shampoo":
                dist_checkpoint.load_state_dict(
                    state_dict=sd,
                    storage_reader=dist_checkpoint.FileSystemReader(f"{self.log_dir}/distcrap")
                )
            if "optim" in sd: self.optimizer.load_distributed_state_dict(sd["optim"], key_to_param=k2p)#self.active_model.named_parameters())



        self.LRsched(self.state.n_step - self.step_offset)
        self.Dropsched(self.state.n_iter)

        for p in self.optimizer.param_groups:
            p['betas'] = self.conf.adam.betas
            if self.conf.optim == 'adam': p['weight_decay'] = self.conf.adam.weight_decay
        self.optimizer.zero_grad()
        
        print(f"Logging every {c.log_every} iterations", only_rank_0=True)
        print(f"Saving every {c.save_every} iterations", only_rank_0=True)
        self.loss = 100
        if self.conf.quit: print("Quitting before training"); exit()
        self.main_timer = LoopTimer()
        ts = False
        self.load_timer = Timer(torch_sync=ts)
        self.gpu_compute_timer = Timer(torch_sync=ts)
        self.step_timer = Timer(torch_sync=ts)
        self.other_timer = Timer(torch_sync=ts)
        self.max_mem = []
        self.max_load_time = []
        self.last_1000_dates = deque(maxlen=1000)
        self.matepoint_size_gb = 0.0

        self.gmag2 = 0
        self.pmag2 = 0
        self.param_max = 0
        self.grad_max = 0

        if not self.conf.lr_sched.schedule_dts:
            self.random_timestep_subset = 0
            self.max_sample_dt = self.data.config.timesteps[-1]
            self.setupDataloader()
        self.load_fn = self.get_load_fn()

        if c.profile:
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=2, warmup=2, active=2, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            )
            self.profiler.start()
        self.memmon = MemoryMonitor()
        print(CYAN("Starting training"), only_rank_0=True)

    ############ TRAINING LOOP ############

    def train(self):
        # Do not add messy shit to this function, add that in the sub-functions. This is so that it's easy to just call parts of it in like profiling
        sample = self.get_sample()
        self.train_compute(sample)
        self.train_step()
        self.train_other()

    def get_sample(self):
        c = self.conf
        assert c.rerun_sample == 1, "no longer implemented after Jan 3rd 2024 cause of refactoring, you need to reimplement it"
        if c.profile: self.profiler.step()
        if self.rank == 0 and c.print_ram_usage: print("\n"+self.memmon.table())
        self.main_timer()
        self.state.epoch = self.state.n_iter // self.N_training_samples
        with self.load_timer:
            if self.conf.lr_sched.schedule_dts:
                self.random_timestep_subset = self.conf.lr_sched.computeNumRandomSubset(self.state.n_step, slow=self.conf.slow_start)
                self.max_sample_dt = self.conf.lr_sched.computeMaxDT(self.state.n_step, self.model.config.processor_dt, slow=self.conf.slow_start)
                if not self.max_sample_dt == self.data.config.timesteps[-1] or self.random_timestep_subset != self.data.config.random_timestep_subset:
                    self.renew_data_loader()
                    print(GREEN(f"Renewed data loader, new max sample dt: {self.max_sample_dt}, random subset: {self.random_timestep_subset}, timesteps: {self.data.config.timesteps}"),only_rank_0=True)
                    self.load_fn = self.get_load_fn()
            sample = self.load_fn()
            if self.conf.start_time is not None:
                print(CYAN(f"Time Till First Sample: {time.time()-self.conf.start_time:0.2f}s "), only_rank_0=True)
                self.conf.start_time = None
                            
        return sample

    def train_compute(self,sample):
        c = self.conf
        # Grab the tensor from the first input and get its shape
        B = sample.inputs[0][1][0].shape[0]
        
        if not c.ignore_train_safegaurd: 
            for d in sample.timestamps:
                assert d <= 1577865600 or d >= 1609488000, f"Bro don't train on the test set: {d}"
        for d in sample.timestamps:
            assert d <= 1722754800, f"bro this is the real real real test set: {d}"

        self.last_1000_dates.append(get_date_str(sample.timestamps[0].item()))
        load_string =  f"step {self.state.n_step}: " + " ".join([get_date_str(d.item()) for d in sample.timestamps])
        with open(self.datesfilepath, "a") as f:
            f.write(load_string + "\n")

        print(f"Doing dts: {sample.dts} \t for input timestamp {snix2date(sample.timestamps[0].item())}")

        with self.gpu_compute_timer:
            if c.diffusion:
                self.compute_all_diffusion(x_gpu, yt_gpus, sample.dts)
                #self.compute_diffusion_combined(x_gpu, yt_gpus, dts)
            else:
                self.compute_all(sample)

        
        self.state.n_iter += B * self.world_size

    def train_step(self):
        c = self.conf
        with self.step_timer:
            print("## Doing Step",only_rank_0=True)
            if self.conf.HALF:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.active_model.parameters(), 4.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            #print(list(self.active_model.named_parameters())[10][1].grad[0].item())
            self.state.n_step+=1 ; self.n_step_since_restart+=1

        print(ORANGE(f" LR {self.current_lr:0.7f} | Loss {self.loss:0.3} | Scaler {self.scaler.get_scale():0.1f} | pmax {self.param_max:.5f} | gmax {self.grad_max:.5f} | pmag {self.pmag2**0.5:.5f} | gmag {self.gmag2**0.5:0.5f} "))


    def train_other(self):
        c = self.conf

        loss_info = f"n_step: {self.state.n_step} | loss: {self.loss} | lr: {self.current_lr} | scaler: {self.scaler.get_scale()} "
        self.lossfile.write(loss_info)
        self.lossfile.flush()
        if self.rank == 0:
            self.writer.add_scalar('Learning/Loss', self.loss, self.state.n_step)
            if self.last_diff is not None:
                self.writer.add_scalar('Learning/DiffLoss', self.last_diff, self.state.n_step)
                #self.writer.add_scalar('Learning/DiffLossRatio', self.last_diff_ratio, self.state.n_step)

            #self.writer.add_scalar(f'Learning/Loss_{dt}', self.Loss[dt], self.state.n_step)
            self.writer.add_scalar('Learning/Rate', self.current_lr, self.state.n_step)
            self.writer.add_scalar('Learning/Drop_Rate', self.current_drop_rate, self.state.n_step)
            self.writer.add_scalar('Learning/GradScaler', self.scaler.get_scale(), self.state.n_step)


        if self.DDP:
            memory_list = [torch.zeros(1).cuda() for _ in range(self.num_gpus)]
            dist.all_gather(memory_list, torch.tensor([torch.cuda.max_memory_reserved() / 1024**3]).cuda())
            max_memory = max(memory.item() for memory in memory_list)
            self.max_mem.append(max_memory)
            load_time_list = [torch.zeros(1).cuda() for _ in range(self.num_gpus)]
            dist.all_gather(load_time_list, torch.tensor([self.load_timer.val]).cuda())
            self.max_load_time.append(max(load.item() for load in load_time_list))
        else:
            self.max_mem.append(torch.cuda.max_memory_reserved() / 1024**3)
            self.max_load_time.append(self.load_timer.val)

        pmag2 = 0
        gmag2 = 0
        gdic = defaultdict(float)
        self.grad_max = 0
        self.param_max = 0
        for name, param in self.active_model.named_parameters():
            #print(name,param.grad)
            if param is None: continue
            self.param_max = max(self.param_max, torch.max(torch.abs(param)).item())
            if param.grad is not None:
                self.grad_max = max(self.grad_max, torch.max(torch.abs(param.grad)).item())
                #print(f"NOTNAN  {torch.isnan(param.grad).sum()} / {param.grad.numel()} {name}")
                #if torch.isnan(param.grad).any():
                    #print(param.grad)
                    #print(f"NANGRAD {torch.isnan(param.grad).sum()} / {param.grad.numel()} {name}", only_rank_0=True)
                sq = (param.grad**2).sum().item()
                if len(name.split(".")) > 1:
                    gdic[name.split(".")[1]] += sq
                gmag2 += sq#allgrads.append(param.grad.cpu().data.flatten())
            pmag2 += (param**2).sum()
        self.lossfile.write(f"| pmag2: {pmag2:.2f} | gmag2: {gmag2:.2f}\n")
        self.lossfile.flush()
        self.pmag2 = pmag2
        self.gmag2 = gmag2

        if self.rank == 0 and (self.state.n_iter % (self.conf.log_step_every) == 0 or self.state.n_step <10):
            #print(f"pmag2: {pmag2**0.5:.2f} | gmag2: {gmag2:.2f}")
            self.writer.add_scalar('Learning/GMag', gmag2**0.5, self.state.n_step)
            for k in gdic:
                self.writer.add_scalar(f'Learning/GMag_{k}', gdic[k]**0.5, self.state.n_step)
            self.writer.add_scalar('Learning/PMag', pmag2**0.5, self.state.n_step)
            self.writer.add_scalar('Learning/Pmax', self.param_max, self.state.n_step)
            self.writer.add_scalar('Learning/Gmax', self.grad_max, self.state.n_step)
            self.writer.add_scalar('Learning/Epoch', self.state.n_iter/self.N_training_samples, self.state.n_step)
            self.writer.add_scalar('Learning/Iteration', self.state.n_iter, self.state.n_step)
            BB = self.world_size
            if self.main_timer.avg != 0: self.writer.add_scalar('Learning/Seconds_per_step', self.main_timer.avg , self.state.n_step)
            if self.main_timer.avg != 0: self.writer.add_scalar('Learning/Samples_per_second_per_gpu', BB / self.main_timer.avg / self.num_gpus , self.state.n_step)
            self.writer.add_scalar('zSystem/Max_VRAM', max(self.max_mem), self.state.n_step)
            self.max_mem = []
            self.writer.add_scalar('zSystem/Max_Load_Time', max(self.max_load_time), self.state.n_step)
            self.max_load_time = []

        self.optimizer.zero_grad(set_to_none=True)
        self.LRsched(self.state.n_step - self.step_offset)
        if self.conf.optim == 'shampoo':
            if self.n_step_since_restart < 20:
                self.optimizer._precondition_frequency = 1
            elif self.n_step_since_restart < 500:
                self.optimizer._precondition_frequency = 5
            else:
                self.optimizer._precondition_frequency = 20

        self.Dropsched(self.state.n_iter)

        with self.other_timer:
            if ((self.state.n_step+1)) % max(c.log_every,1) == 0:
                check = model_checksum(self.active_model)
                print('model checksum:',check)

                #self.writer.add_scalar('Learning/Loader_dt', avg_dataloader_dt, self.state.n_step)

                self.writer.add_scalar('Learning/Epoch', self.state.n_iter/self.N_training_samples, self.state.n_step)
                print(f"Epoch {self.state.n_iter // self.N_training_samples}|",time.strftime("%H:%M:%S", time.localtime()))
                self.writer.add_scalar('Conf/num_gpus', self.num_gpus , self.state.n_step)

                if self.rank == 0:

                    max_count = max(Counter(self.last_1000_dates).values())
                    self.writer.add_scalar('zDataloader/max dups last 1000', max_count , self.state.n_step)
                    ram = self.memmon.get_main_rss() / 1024**3
                    self.writer.add_scalar('zSystem/Main Process RAM', ram , self.state.n_step)
                    self.writer.add_scalar('zDataloader/random_timestep_subset', self.random_timestep_subset , self.state.n_step)
                    self.writer.add_scalar('zDataloader/max_sample_dt', self.max_sample_dt , self.state.n_step)

            
            
            if (self.state.n_step+1) % c.save_every == 0: 
                with self.main_timer.pause():
                    try: 
                        self.save_weights()
                    except Exception as e:
                        print("Oh jesus, saving is broken, I guess this run is for the leaderboards and nothing else, o7")
                        print(e)
                        traceback.print_exc()

            print(MAGENTA(f"step {self.state.n_step} | Load: {self.load_timer.val:.2f} | GPU: {self.gpu_compute_timer.val:.2f} | Step: {self.step_timer.val:.2f} | Other: {self.other_timer.val:.2f} | Total: {self.main_timer.get():.2f} | VRAM {torch.cuda.max_memory_reserved() / 1024**3:.2f}GiB | RAM {self.memmon.get_main_rss() / 1024**3:.2f}GiB | MATE {self.matepoint_size_gb:.2f}GiB"))

    def Dropsched(self,iter):
        if not self.conf.drop_path:
            self.current_drop_rate = 0
            return
        c = self.conf.drop_sched
        drop = self.computeDrop(iter,c)
        print('drop rate:',drop)
        try: self.active_model.module.change_droppath_prob(drop)
        except:
            print("uhhhhhhh couldn't set drop rate?????")
            assert "validate" in sys.argv[0]

    @staticmethod
    def computeDrop(iter,c):
        print('[drop] iter',iter,c.iter_offset)
        iter = max(iter + c.iter_offset,0)
        drop = np.interp(iter, [0, c.ramp_length], [c.drop_min, c.drop_max])
        if iter > c.ramp_length:
            drop = c.drop_max
        return drop 
            
    def print_mem(self,s):
        print_mem(s,dev=self.primary_compute)

    # Gathers the weights for the loss function at dt
    def gather_dt_loss_weight(self, dt):
        if dt not in self.dt_loss_weights:
            if dt in self.conf.dt_loss_weights_override:
                self.dt_loss_weights[dt] = self.conf.dt_loss_weights_override[dt]
            else:
                dt_loss_weight = 1/((dt/24)**(3**0.5)) if dt != 0 else 0.5
                self.dt_loss_weights[dt] = min(dt_loss_weight, 2.0)

        if dt in self.conf.dt_loss_weights_override:
            self.dt_loss_weights[dt] = self.dt_loss_weights[dt] * self.conf.dt_loss_beta + self.conf.dt_loss_weights_override[dt] * (1-self.conf.dt_loss_beta)

        return self.dt_loss_weights[dt]

    def compute_all(self,sample):
        total_loss = self.compute_forward(sample)
        self.compute_backward(total_loss)

    def compute_backward(self,total_loss):
        print("## Doing Backwards",only_rank_0=True)
        if self.model.config.nsight: torch.cuda.nvtx.range_push("backwards")
        self.compute_backward_inner(total_loss)
        if self.model.config.nsight: torch.cuda.nvtx.range_pop()

    @TIMEIT(sync=True)
    def compute_forward(self,sample):
        # Gather relevant data for each encoder / decoder 
        x = sample.get_x_t0(self.model.encoders)
        assert not sum([torch.isnan(elem).any() for elem in x]), f"NAN in x, {x[-1]}"
        yt = sample.get_y_ts(self.model.decoders)
        additional_inputs = dict(sample.get_additional_inputs()) # turns defaultdict into dict; DDP does not like defaultdict
        dts = sample.dts
        assert len(yt) == len(dts)
        self.model.config.name = self.conf.name



        x_gpu = to_device(x, self.primary_compute)    
        yt_gpus = to_device(yt, self.primary_compute, non_blocking=True)

        print("## Doing Forwards",only_rank_0=True)

        with torch.autocast(enabled=bool(self.conf.HALF), device_type='cuda', dtype=torch.float16):
            y_gpus = self.active_model(x_gpu, dts, additional_inputs=additional_inputs)

        total_loss = 0
        if "latent_l2" in y_gpus:
            self.writer.add_scalar('Learning/LatentL2', y_gpus["latent_l2"], self.state.n_step)
            total_loss += self.conf.latent_l2 * y_gpus["latent_l2"]
            print("added latent l2", total_loss.item(), "raw", y_gpus["latent_l2"].item())
        if self.model.config.nsight: torch.cuda.nvtx.range_push("forwards")
        
        def filt_y_len(i):
            y_gpu = y_gpus[dts[i]]
            if len(y_gpu) == 0: # Edge case related to skipping loss for a particular decoder (used in Regional TC decoding)
                print(ORANGE(f"Skipping decoder {decoder.__class__.__name__}..."))
                assert self.conf.find_unused_params, "Skipping a decoder is only supported with find_unused_params=True (you'll error out anyways with it as False because some model params aren't being used)"
                return False
            return True

        # i is dt, j is decoder
        @TIMEIT(sync=True)
        def compute_loss_decoder(i, j, dt_loss_weight):
            dt = dts[i]
            y_gpu, yt_gpu, decoder = y_gpus[dt][j], yt_gpus[i][j], self.model.decoders[j]
            any_nan = torch.isnan(y_gpu).any()
            if any_nan:
                print("uhhhhh nans!!")
                red = list(torch.sum(torch.isnan(y_gpu,axis=(0,1,2))).cpu().numpy())
                print(red)
                for a, b in enumerate(red):
                    if b != 0: print("aaaaa index", a, b)
            assert not any_nan, f"NAN in y_gpu!! {torch.sum(torch.isnan(y_gpu))} / {y_gpu.numel()} (num nans / total). y_gpu.shape: {y_gpu.shape}. This is model output, so something inputted into the model is fishy"

            if decoder.__class__.__name__ in POINT_DECODERS:
                yt_full = yt_gpu #heinous
                yt_gpu = torch.concat([y['data'] for y in yt_gpu])
            # Compute nan mask since we don't want nans to backprop
            nan_mask = ~torch.isnan(yt_gpu)
            y_gpu = y_gpu * nan_mask
            yt_gpu = torch.nan_to_num(yt_gpu, nan=0)
            
            assert not torch.isnan(yt_gpu).any(), f"NAN in yt_gpu!! {torch.sum(torch.isnan(yt_gpu))} / {yt_gpu.numel()} (num nans / total). yt_gpu.shape: {yt_gpu.shape}"

            args = [y_gpu, yt_gpu]
            if decoder.__class__.__name__ in POINT_DECODERS:
                args.append(yt_full)
            with torch.autocast(enabled=bool(self.conf.HALF), device_type='cuda', dtype=torch.float16):
                decoder_loss = decoder.compute_loss(*args)
            
            # Delta Time Weight Handling
            # Check Decoder in decoder.py for more of an explanation
            loss = decoder_loss * dt_loss_weight
            
            # Loss handling
            if torch.isnan(loss): print("NAN LOSS", dt, decoder.__class__.__name__)

            if self.rank == 0:
                self.writer.add_scalar(f'Learning/Loss_{dt}_{decoder.__class__.__name__}', loss.item(), self.state.n_step)

            return loss
        
        def compute_loss_dt(i):
            dt, y_gpu = dts[i], y_gpus[dts[i]]
            dt_loss_weight = self.gather_dt_loss_weight(dt)
            loss = sum(map(lambda j: compute_loss_decoder(i, j, dt_loss_weight), filter(filt_y_len, range(len(y_gpu)))))
            if self.rank == 0:
                self.writer.add_scalar(f'Learning/Loss_{dt}', loss.item(), self.state.n_step)
            return loss

        total_loss += sum(map(compute_loss_dt, range(len(dts))))
        self.loss = total_loss.item()

        if self.state.n_step % max(self.conf.log_every, 1) == 0: 
            def log_decoder(i,j):
                dt = dts[i]
                y_gpu, yt_gpu, decoder = y_gpus[dt][j], yt_gpus[i][j], self.model.decoders[j]
                kwargs = {"y_gpu": y_gpu, "yt_gpu": yt_gpu, "trainer": self}
                rms_dict = decoder.compute_errors(**kwargs)
                assert self.conf.save_imgs_every % self.conf.log_every == 0, "save_imgs_every must be a multiple of log_every otherwise you will be sad"
                if self.rank == 0:
                    img_dir = os.path.join(self.log_dir, "imgs") if self.state.n_step % self.conf.save_imgs_every == 0 else None    
                    prefix = f"Error{'_decoder' + str(j + 1) if len(self.model.decoders) > 1 else ''}"
                    decoder.log_information(y_gpu, yt_gpu, rms_dict, self.writer, dt, self.state.n_step, prefix, img_dir)
                return rms_dict
            with torch.no_grad():
                rms_all = { dts[i]: { j: log_decoder(i, j) for j in range(len(y_gpus[dts[i]])) } for i in range(len(dts))}

                if self.conf.do_da_compare:
                    da_compare(self, x_gpu, dts, yt_gpus, rms_all)


        # The below lines are for good measure now, ut should not be needed, given that the function now ends before the backwards pass.
        del x_gpu; 
        del y_gpus; 
        del yt_gpus; 


        if self.model.config.nsight: torch.cuda.nvtx.range_pop()
        if self.DDP:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss /= dist.get_world_size()
        if self.rank == 0:
            #print(f"dt_loss_weights: {self.dt_loss_weights}")
            for dt in self.dt_loss_weights:
                self.writer.add_scalar(f"zLossWeights/loss_weights[{dt}]", self.dt_loss_weights[dt], self.state.n_step)
            
        # pinned memory from matepoint is allways rounded up to the nearest power of 2
        def round_up_pow2(n):
            return 1 << (n - 1).bit_length()

        if 0:
            # Note: this should be handled more cleanly in the new matepoint library
            if self.DDP: matepoints = self.active_model.module.get_matepoint_tensors()
            else: matepoints = self.active_model.get_matepoint_tensors()
            self.matepoint_size_gb = sum([round_up_pow2(x.numel() * x.element_size()) for x in matepoints]) / 1024**3
        else:
            self.matepoint_size_gb = -1

        return total_loss

    @TIMEIT(sync=True)
    def compute_backward_inner(self,loss):
        if self.conf.HALF:
            sl = self.scaler.scale(loss)

            if self.state.n_step % self.conf.compute_Bcrit_every == 0 and self.DDP:
                t0 = time.time()
                params = sorted(list(self.active_model.named_parameters()))
                gg = torch.autograd.grad(sl, [x[1] for x in params], retain_graph=True)

                """
                t0x = time.time()
                if self.rank == 0: l = [None for _ in range(self.world_size)]
                else: l = None
                torch.distributed.gather_object(gg, object_gather_list=l, dst=0)
                print("fastgather took", time.time()-t0x)
                """
                TrS = defaultdict(float)
                SG2 = defaultdict(float)

                means = {}

                t0g = time.time()
                for i in range(len(params)):
                    p = params[i][0]
                    if self.rank == 0:
                        l = [torch.zeros_like(gg[i]) for _ in range(self.world_size)]
                    else:
                        l = None
                    torch.distributed.gather(gg[i], gather_list=l, dst=0)
                    if self.rank == 0:
                        l = torch.stack(l, dim=0)
                        #print(l)
                        var, mean = torch.var_mean(l, axis=0)
                        #print("var shape", var.shape, mean.shape)
                        means[p] = mean
                        trS = var.sum().item()
                        #print("uhh", trS.shape, trS.dtype, trS.device, "vs", gg[i].shape)
                        sG2 = torch.sum(torch.square(mean)).item()
                        TrS["overall"] += trS
                        SG2["overall"] += sG2

                        which = p.split(".")[1]
                        TrS[which] += trS
                        SG2[which] += sG2
                        del var
                        del l

                    else:
                        means[p] = gg[i] * np.nan
                    means[p] = means[p].contiguous()
                    torch.distributed.broadcast(means[p], src=0) 
                print("gather took", time.time()-t0g)

                del gg

                if self.rank == 0:
                    for k in TrS:
                        print("Bsimple", k, TrS[k]/SG2[k])
                        self.writer.add_scalar("MetaLearning/Bsimple_"+k, TrS[k]/SG2[k], self.state.n_step)


                for pn, p in params:
                    #assert not torch.isnan(means[pn]).any()
                    p.grad = means[pn]
                #for (p, _), g in zip(params, gg):
                #    print(p, torch.sqrt(torch.mean(torch.square(g))), g.dtype, g.shape)
                #print("yooo", gg, len(gg), len(params))
                print("initial autograd took", time.time()-t0)
            else:
                sl.backward()

        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.active_model.parameters(), 2.0)


    @TIMEIT()
    def get_load_fn(self):
        dl_iter = iter(self.data_loader)
        def load_fn():
            nonlocal dl_iter
            try:
                return next(dl_iter)
            except StopIteration:
                dl_iter = iter(self.data_loader)
                return next(dl_iter)
        return load_fn

    def renew_data_loader(self):
        pdt = self.model.config.processor_dt
        assert len(pdt) == 1, "need to update this for multiple procs"
        del self.data_loader
        config = self.data.config
        config.timesteps = list(range(0,self.max_sample_dt+1,pdt[0]))
        config.random_timestep_subset = self.random_timestep_subset
        config.update()
        del self.data
        self.data = WeatherDataset(config)
        self.setupDataloader()

    def setupDataloader(self):
        self.data.check_for_dates()
        manager = mp.Manager()
        pid_dict = manager.dict()
        init_fn = partial(worker_init_fn, pid_dict=pid_dict)
        self.data_loader = DataLoader(self.data, batch_size=self.conf.batch_size, shuffle=True, num_workers=self.conf.num_workers, 
                                      prefetch_factor=self.conf.prefetch_factor, pin_memory=self.conf.pin_memory, worker_init_fn=init_fn,
                                      collate_fn=collate_fn)
        self.data_loader.pid_dict = pid_dict    
        self.N_training_samples = len(self.data_loader)

    @staticmethod
    def robust_gather(obj,rank,world_size,num_retry):
        if rank == 0:
            for i in range(num_retry):
                try:
                    gather = [None]*world_size
                    dist.gather_object(obj,gather)
                    return gather
                except Exception as e:
                    print(f"Rank {rank}: Gather {i} failed, retrying. {e}")
                    traceback.print_exc()
                    time.sleep(0.1)
        for i in range(num_retry):
            try:
                dist.gather_object(obj,dst=0)
                return None
            except Exception as e:
                print(f"Rank {rank}: Gather {i} failed, retrying. {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def get_buffers_checksum(self,_print=True):
        buf_checksum = sha256()
        s = 0
        for name, attr in self.model.named_buffers():
            #print(f"{name}: shape={attr.shape}, dtype={attr.dtype}, type=Buffer")
            tensor_bytes = attr.cpu().numpy().tobytes()
            s+=len(tensor_bytes)
            buf_checksum.update(tensor_bytes)
        if _print:
            print("buffers checksum", buf_checksum.hexdigest())
            print("total buffer size", s/1e6, "MB")
        return buf_checksum.hexdigest()

    def save_weights(self,keep=False):
        print(GREEN("Saving weights"))
        if os.path.exists(f'{self.log_dir}/joank.py'):
            # widely regarded as the biggest innovation in PL theory since goto's
            try:
                with open(f'{self.log_dir}/joank.py') as f:   # I'm sorry joan, but I don't want ur code in my runs
                    t = 'if 1:\n'+f.read()
                exec(t)
            except:
                print("uhhh wild attempt failed, sad")
                traceback.print_exc()
        if self.conf.optim == 'shampoo' and self.conf.save_optimizer:
            try:
                state_dict = {"optim": self.optimizer.distributed_state_dict(key_to_param=self.k2p)}
                os.makedirs(self.log_dir+"/distcrap", exist_ok=True)
                dist_checkpoint.save_state_dict(
                    state_dict=state_dict,
                    storage_writer=dist_checkpoint.FileSystemWriter(self.log_dir+"/distcrap"),
                )
            except:
                print("Ugh, shampoo save failed, sad")
                traceback.print_exc()
        if not self.rank == 0: return
        try:
            buffer = save_model_preforked(self)
        except:
            print("Ugh, shampoo save failed, sad")
            traceback.print_exc()

        if self.last_pid is not None:
            os.waitpid(self.last_pid,0)

        self.last_pid = os.fork()
        if self.last_pid != 0:
            return

        # the promised land
        try:
            save_weights(self,buffer,keep=keep)
        except (KeyboardInterrupt,Exception) as e: 
            print(RED("[Save] Fork failed" + str(e)))
            os._exit(1)

        os._exit(0)

    def audit_inputs(self,xs,do_print=True):
        if (not self.do_audit) and (self.audit_stats.checked_inputs): return
        if self.conf.skip_audit: return 
        errstr = ""
        print("auditing??")
        for i,x in enumerate(xs[:-1]):
            mesh = self.data.inputs[i]
            B,N1,N2,D = x.shape 
            #print("uhhh shape is", B, N1, N2, D)
            #print("meshstuff", mesh.extra_sfc_vars, mesh.extra_sfc_pad)
            with torch.no_grad():
                xp = x[...,:mesh.n_pr].reshape(B,N1,N2,mesh.n_pr_vars,mesh.n_levels)
                for i, var in enumerate(mesh.pressure_vars):
                    mean, std = torch.mean(xp[:,:,:,i,:]), torch.std(xp[:,:,:,i,:])
                    if do_print: print(f'{var:<10} mean {mean:<10.3f} std {std:<10.3f}')
                    if not -1 < mean < 1: errstr+= f"{var} mean is sus {mean}\n"
                    if var == "zeropad":
                        if std > 0.001: errstr += "what the fuck zeropad mean,std is {mean}, {std}\n"
                    else:
                        if not 0.4 < std < 2: errstr+= f"{var} std is sus {std}\n" 
                
                xo = x[...,mesh.n_pr:].reshape(B,N1,N2,-1)

                AMNESTY = ["45_tcc"]#, "logtp", "15_msnswrf"]

                for i,var in enumerate(mesh.sfc_vars):
                    if i >= 4 and do_print: print("Sfc var", i, xo.shape, xo[..., i])
                    mean, std = torch.mean(xo[:,:,:,i]), torch.std(xo[:,:,:,i])
                    if do_print: print(f'{var:<10} mean {mean:<10.3f} std {std:<10.3f}')
                    if not -1 < mean < 1: errstr+= f"{var} mean is sus {mean}\n"
                    #if not 0.4 < std < 1.8 and var not in AMNESTY: errstr+= f"{var} std is sus {std}\n" 
                    if var == "zeropad":
                        if std > 0.001: errstr += f"what the fuck zeropad mean,std is {mean}, {std}\n"
                    else:
                        if not 0.4 < std < 2 and var not in AMNESTY: errstr+= f"{var} std is sus {std}\n" 
        assert errstr == "", "Audit failed:\n" + errstr
        print('done auditing')
        self.audit_stats.checked_inputs = True

def get_trainer(args):
    mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
    w = WeatherTrainer(args,mesh,None)
    w.setupData()
    return w

import hashlib
def model_checksum(model):

    hasher = hashlib.sha256()
    
    for param in model.parameters():
        param_data = param.data.cpu().numpy().tobytes()
        hasher.update(param_data)
    
    return hasher.hexdigest()
