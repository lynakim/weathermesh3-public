import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from data import *
from model_latlon_3d import *
from utils import *
from train import *
import json
import GPUtil


OUTPATH = 'ignored/perf/jsons'
os.makedirs(OUTPATH,exist_ok=True)

DTS = [6,12]
TORCH_PROF = False
TORCH_MEM = True


NSIGHT = 'NSYS_TARGET_STD_REDIRECT_ACTION' in os.environ
if TORCH_PROF:
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
        #on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )

self = None

def get_self(data):
    global self
    if self is None:
        self = WeatherTrainer(data)
        self.primary_compute = torch.device('cuda:0')
        #self.load_matrices()
        self.scaler = torch.cuda.amp.GradScaler()
        self.rank = 1
        self.conf.log_every = 1000000
        self.conf.save_imgs_every = 1000000
        self.conf.save_every = 1000000
        self.Loss = {}
        self.max_mem = []
        self.max_load_time = []
        ts = False
        self.load_timer = Timer(torch_sync=ts)
        self.gpu_tx_timer = Timer(torch_sync=ts)
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

    return self


def test(model,data,n=4):
    self = get_self(data)
    timers = []
    timer = Timer(name="gpu_time",torch_sync=True,nsight="forwards"); timers.append(timer)

    mems = []
    mem = GPUMemoryMonitor(name="gpu_mem")
    model.config.nsight = NSIGHT
    self.model = model
    self.active_model = model.to(self.primary_compute)
    self.writer = SummaryWriter('/tmp/')
    self.state_norm_matrix_gpu = self.model.decoders[0].state_norm_matrix.to(self.primary_compute).detach()
    self.optimizer = torch.optim.AdamW(self.active_model.parameters(), lr=1e-3)
    results = {}
    def RECORD(x,y):
        if not x in results:
            results[x] = [y] 
        else:
            results[x] += [y]

    if TORCH_PROF: prof.start()
    for i in range(n):
        torch.cuda.empty_cache()
        if TORCH_PROF: prof.step()
        if TORCH_MEM:  
            torch.cuda.memory._record_memory_history()

        with Timer(name="sample_time",print=False):
            sample = collate_fn([data[i]])
        
        if NSIGHT: torch.cuda.cudart().cudaProfilerStart()
        if NSIGHT: torch.cuda.nvtx.range_push("iteration{}".format(i))

        x_gpu = [x_.to(self.primary_compute, non_blocking=True) for x_ in sample.get_x_t0()]
        yt_gpu = [y_[1][0].to(self.primary_compute, non_blocking=True) for y_ in sample.outputs]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with timer, mem:
            self.compute_all(x_gpu,yt_gpu,DTS)
            self.model_step(1)
        #if NSIGHT: torch.cuda.nvtx.range_pop()
        gpu = GPUtil.getGPUs()[0]
        RECORD("gpu_mem",gpu.memoryUsed)
        for t in timers: RECORD(t.name,t.val)
        for m in mems: 
            for v in ['alloced','reserved']:
                RECORD(m.name+'_'+v[0],m.__dict__[v])
        st='dt: '+str(DTS)+' '
        for k,v in results.items():
            st+=f"{k}: {v[-1]:<3.2f} "
        print(st)
        if NSIGHT: torch.cuda.nvtx.range_pop()
    if TORCH_PROF: 
        prof.stop()
        prof.export_memory_timeline('ignored/perf/memory_timeline.html')
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.memory._dump_snapshot("ignored/perf/memory_snapshot.pickle")
    return results

def get_Dec8():
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    data = WeatherDataset(DataConfig(inputs=[imesh], outputs=[omesh],
                                            timesteps=DTS,
                                            requested_dates = tdates
                                            ))
    import model_latlon.top as top
    from model_latlon.encoder import SimpleConvEncoder 
    conf = top.ForecastModelConfig(
        [imesh],
        latent_size = 1280,
        patch_size = (4,8,8),
        affine = True,
        pr_depth = [8],
        encdec_tr_depth = 2,
        oldpr = False,
        tr_embedding = 'rotary', 
    )
    encoder = SimpleConvEncoder(imesh,conf)
    decoder = top.StackedConvPlusDecoder(omesh,conf)
    model = top.ForecastModel(imesh,conf,encoders=[encoder],decoders=[decoder])
    print(model)
    data.check_for_dates()
    return model, data


print(test(*get_Dec8()))