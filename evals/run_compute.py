import os 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from data import *
from model_latlon_3d import *
from utils import *
from train import *
import json
import GPUtil

#
# FYI
# This code is super old. this file will not work anymore but leaving it here for reference
# Look at the profile folder, that's where I've moved all of this stuff to.


OUTPATH = 'ignored/perf/jsons'
os.makedirs(OUTPATH,exist_ok=True)

DT = 24

extra = ['logtp', '15_msnswrf', '45_tcc']
mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
data = WeatherDataset(DataConfig(inputs=[mesh],
                                        timesteps=[DT],
                                        requested_dates = get_dates((D(1997, 9, 1),D(1997, 9,3))),
                                        ))
data.check_for_dates()
self = WeatherTrainer(data)
self.primary_compute = torch.device('cuda:0')
self.load_matrices()
self.scaler = torch.cuda.amp.GradScaler()
self.rank = 1
self.conf.log_every = 1000000
self.conf.save_imgs_every = 1000000
self.conf.save_every = 1000000
self.Loss = {}


#def get_model(mesh,):
#    model = ForecastStepSwin3D(ForecastStepConfig(mesh))

NSIGHT = False

def hash_gradients(model):
    hash_obj = hashlib.sha256()
    for param in model.parameters():
        if param.grad is not None:
            grad_bytes = param.grad.cpu().numpy().tobytes()
            hash_obj.update(grad_bytes)
    return hash_obj.hexdigest()


def shash(tensor):
    tensor_bytes = tensor.to('cpu').detach().numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

def test(model,n=3):
    timers = []
    timer = Timer(name="gpu_time",torch_sync=True,nsight="forwards"); timers.append(timer)

    mems = []
    mem = GPUMemoryMonitor(name="gpu_mem")

    self.model = model
    self.active_model = model.to(self.primary_compute)
    self.state_norm_matrix_gpu = torch.from_numpy(self.state_norm_matrix).to(self.primary_compute).detach()
    results = {}
    def RECORD(x,y):
        if not x in results:
            results[x] = [y] 
        else:
            results[x] += [y]

    for i in range(n):
        torch.cuda.empty_cache()
        with Timer(name="sample_time",print=False):
            sample = default_collate([data[i]])
        x = sample[0]
        #print("input hash:",shash(x))
        #print(x.numpy().astype(np.float64).sum())
        yt = sample[1]

        if NSIGHT: torch.cuda.cudart().cudaProfilerStart()
        if NSIGHT: torch.cuda.nvtx.range_push("iteration{}".format(i))

        x_gpu = [x_.to(self.primary_compute, non_blocking=True) for x_ in x]
        yt_gpu = yt[0].to(self.primary_compute, non_blocking=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with timer, mem:
            if NSIGHT: torch.cuda.nvtx.range_push("forwards")
            self.compute_all(x_gpu,[yt_gpu],[DT])
        #print("fw hash:",shash(y_gpu))
        #if NSIGHT: torch.cuda.nvtx.range_pop()
        #loss = self.compute_loss(x_gpu[0],y_gpu,yt_gpu[0],x[0].shape,dt)
        #with backwards_timer, backwards_mem:
        #    self.compute_backward(loss)
        if NSIGHT: torch.cuda.nvtx.range_pop()
        #print("bk hash:",hash_gradients(self.active_model))
        gpu = GPUtil.getGPUs()[0]
        RECORD("gpu_mem",gpu.memoryUsed)
        for t in timers: RECORD(t.name,t.val)
        for m in mems: 
            for v in ['alloced','reserved']:
                RECORD(m.name+'_'+v[0],m.__dict__[v])
        st='dt: '+str(DT)+' '
        for k,v in results.items():
            st+=f"{k}: {v[-1]:<3.2f} "
        print(st)
        if NSIGHT: torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
     
    return results


def run(setting,range_,fix={}):
    for val in range_:
        print(f'{setting}={val}')
        torch.cuda.empty_cache()
        time.sleep(0.1)
        confd = {
            "hidden_dim":1120,
            "dims_per_head":40,
            "patch_size":(4,8,8),
            "window_size":(5,5,7),
            }
        confd.update(fix)
        if '[' not in setting: 
            confd[setting] = val
        conf = ForecastStepConfig([mesh],**confd)
        if '[' in setting:
            idx = int(setting.split('[')[1].split(']')[0])
            nval = list(conf.__dict__[setting.split('[')[0]])
            nval[idx] = val
            conf.__dict__[setting.split('[')[0]] = tuple(nval)
        conf.update()
        model = ForecastStep3D(conf)
        results = test(model)
        results["conf"] = confd
        #results["config_dict"] = conf.__dict__
        with open(f'{OUTPATH}/{setting}={val}.json','w') as f:
            json.dump(results,f,indent=2)
        del model
        del self.active_model
        del self.model


if __name__ == '__main__':
    from evals.plot_compute import plot_all 
    #run('proc_swin_depth',range(4,64,4))
    #run('hidden_dim',range(512, 896 , 32))
    #run('patch_size[1]',[4,6,8,10,12])
    #run('patch_size[2]',[4,6,8,10,12])
    #run('checkpoint_every',[1,2,3,4,5,6,7,8],fix={'hidden_dim':128})
    #run('dims_per_head',[8,16,32,64,128])
    #run('window_size[2]',[3,6,9,12,15,18,30]
    
    #run('proc_swin_depth',[24,26])
    #run('dec_swin_depth',[0,2,4,6])
    #plot_all()

    #run('window_size[2]',[3,5,7,9,11,15,19])
    #run('window_size[0]',[3,5])
    run('dims_per_head',[40])





    