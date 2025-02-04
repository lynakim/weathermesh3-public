from utils import *
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_latlon.data import get_constant_vars, get_additional_vars, N_ADDL_VARS
from model_latlon.codec2d import EarthConvEncoder2d, EarthConvDecoder2d
from model_latlon.codec3d import EarthConvEncoder3d, EarthConvDecoder3d
from model_latlon.transformer3d import SlideLayers3D, posemb_sincos_3d, add_posemb, tr_pad, tr_unpad
from model_latlon.primatives2d import southpole_pad2d, call_checkpointed, print_total_params, southpole_unpad2d, matepoint
from model_latlon.primatives3d import southpole_pad3d, southpole_unpad3d, earth_pad3d
from model_latlon.decoder import *
from meshes import LatLonGrid
from model_latlon.process import simple_gen_todo
try:
    import healpy as hp
except:
    print("davy ragland is a true scholar")
    import os
    os.system("pip3 install healpy")
    import healpy as hp

from model_latlon.config import ForecastModelConfig

class EnsembleForecastModel(nn.Module,SourceCodeLogger):
    def __init__(self, config, encoders=[], decoders=[], processors=None):
        super(EnsembleForecastModel, self).__init__()
        self.code_gen = 'gen3'
        self.config = config
        
        self.embedding_module = config.embedding_module
        self.encoders = nn.ModuleList(encoders)
        if processors is None:
            processors = {}
            for i,dt in enumerate(self.config.processor_dt):
                processors[str(dt)] = SlideLayers3D(dim=self.config.latent_size, depth=config.pr_depth[i], num_heads=config.num_heads, window_size=self.config.window_size, embedding_module=self.embedding_module, checkpoint_type=self.config.checkpoint_type)

        self.processors = nn.ModuleDict(processors)
        self.decoders = nn.ModuleList(decoders)

        if isinstance(self.config.outputs, list): assert isinstance(self.config.outputs[0], LatLonGrid), f"Code only works right now if LatLonGrid is the first mesh type"

        def ensfn(h):
            return 4

        def carryfn(h):
            return 2
        self.num_ensemble_members = ensfn
        self.num_carryover = carryfn

        self.degs_noise = np.logspace(0.3, 1.5, 32)

        if not config.ens_nomean: self.mean_proj = nn.Linear(config.latent_size, config.latent_size)
        self.std_proj = nn.Linear(config.latent_size, config.latent_size)
        self.noise_proj = nn.Linear(len(self.degs_noise), config.latent_size)

        print(MAGENTA(f"Initializing model. model.config.checkpoint_type is 👉{self.config.checkpoint_type}👈 Is this what you want?"))

    def forward_inner(self, xs, todo_dict, send_to_cpu, callback, return_latents=False, **kwargs):
        # TODO: Clean this up to use the stuff in process.py.

        c = self.config

        if type(xs) == list:
            t0s = xs[-1]
            xs = xs[0:-1]
        else:
            t0s = None

        def encode(xs,t0s):
            return self.encoders[0](xs[0],t0s) 
        
        # Time is in to_unix format (seconds since 1970)
        def decode(x, time, output_idx):
            outputs = []
            for dec in self.decoders:
                outputs.append(dec(x))
            return outputs
        
        todos = [SimpleNamespace(
                target_dt=k,
                remaining_steps = v.split(','),
                completed_steps = [],
                state = (xs if v.startswith('E') else xs[0],None), # in case we go straight from latent
                accum_dt=0,
                output_idx=i, # 0 is the first output
            ) for i, (k,v) in enumerate(todo_dict.items())]

        total_l2 = 0
        total_l2_n = 0
        outputs = {}
        def latent_l2(aa):
            return torch.mean(aa**2)
            return call_checkpointed(lambda a: torch.mean(a**2), aa)
        while todos:
            tnow = todos[0]
            """
            torch.cuda.empty_cache()
            mem = torch.cuda.mem_get_info()
            if int(os.environ.get('RANK', 0)) == 0: print("doing", tnow.remaining_steps[0], mem[0]/1e9, mem[1]/1e9)
            """
            #_t0 = time.time()
            step = tnow.remaining_steps[0]
            completed_steps = tnow.completed_steps.copy()
            x,extra = tnow.state
            accumulated_dt = 0
            #for todo in todos:
            #    print(f"{todo.target_dt}: {','.join(todo.remaining_steps)}")
            if step == 'E':
                x = encode([xx.clone() for xx in x],t0s+3600*tnow.accum_dt)
                total_l2 += latent_l2(x); total_l2_n += 1
            elif step == 'rE':
                assert c.rollout_reencoder, "rollout reencoder not enabled"
                x = self.rollout_reencoder(x[0],t0s+3600*tnow.accum_dt)
            elif step.startswith('P'):
                pdt = int(step[1:])
                x = self.processors[str(pdt)](x)
                #pdt = c.processor_dt[0]
                #assert len(c.processor_dt) == 1, "only one processor_dt is supported"
                #x = self.processor(x)
                accumulated_dt += pdt
            elif step == 'D':
                total_l2 += latent_l2(x); total_l2_n += 1
                dth = tnow.target_dt
                nens = self.num_ensemble_members(dth)
                ncarry = self.num_carryover(dth)

                D = len(self.degs_noise)
                E = nens
                NPIX = 12288

                t0 = time.time()
                data = np.random.randn(E, D, NPIX).astype(np.float32)
                out = np.zeros_like(data,dtype=np.float16)
                for j in range(E):
                    for i, fwhm in enumerate(self.degs_noise):
                        out[j,i] = hp.sphtfunc.smoothing(data[j,i], fwhm=fwhm*np.pi/180, nest=True, iter=1)
                nvertical = x.shape[2]
                #print("yooo nvertical is", nvertical)
                out2 = torch.from_numpy(out).to(x.device).permute(0,2,1)[:,:, None, :].repeat_interleave(nvertical, dim=2)
                #print("out is", out.shape, "after massage is", out2)
                noise = self.noise_proj(out2)

                carried = (list(range(x.shape[0]))*nens)[:nens]
                def stuff(xx, cc, nn):
                    #print("uhhh", xx.shape, nn.shape, cc)
                    if not c.ens_nomean:
                        xm = self.mean_proj(xx)
                    else:
                        xm = xx
                    sm = self.std_proj(xx)
                    return xm[cc] + sm[cc] * nn * 0.25

                xn = call_checkpointed(stuff, x.clone(), carried, noise, checkpoint_type=self.config.checkpoint_type)


                x = decode(xn, (t0s + ( tnow.accum_dt * 60 * 60)).item(), tnow.output_idx)

                carry2 = random.sample(range(nens), ncarry)
                xn = xn[carry2]
                for t in todos[1:]:
                    if t.remaining_steps[0].startswith('P'):
                        t.state = (xn, None)

            else:
                assert False, f"Unknown step {step}"
            for todo in todos:
                if todo.remaining_steps[0] == step and todo.completed_steps == completed_steps:
                    todo.state = (x,extra)
                    todo.accum_dt += accumulated_dt
                    todo.remaining_steps = todo.remaining_steps[1:]
                    todo.completed_steps.append(step)
                    if not todo.remaining_steps:
                        if callback is not None:
                            callback(todo.target_dt, x)
                        else:
                            outputs[todo.target_dt] = x
                        todos.remove(todo)
            #torch.cuda.synchronize()
            #if int(os.environ.get('RANK', 0)) == 0: print("took", time.time()-_t0)
        outputs["latent_l2"] = total_l2 / total_l2_n
        return outputs
    
    def forward(self, xs, todo, send_to_cpu=False, callback=None, return_latents=False, from_latent=False, clear_matepoint=True, **kwargs):
        if hasattr(self.config, 'hmesh') and self.config.hmesh.g2m_idx.device != xs[0].device:
            print("moving my shit over! john don't look")
            self.config.hmesh = self.config.hmesh.to(xs[0].device)
            self.config.hlmesh = self.config.hlmesh.to(xs[0].device)
        if clear_matepoint: matepoint.Gmatepoint_ctx = []
        def is_list_of_ints(obj): return isinstance(obj, list) and all(isinstance(x, int) for x in obj)
        if is_list_of_ints(todo):
            todo = simple_gen_todo(sorted(todo),self.config.processor_dt, from_latent=from_latent)
        if len(self.decoders) > 1 and isinstance(self.decoders[1], PointDecoder):
            additional_inputs = kwargs.get('additional_inputs', {})
            assert 'points' in additional_inputs, "point_data must be passed in for PointDecoder"
            assert 'timestamps' in additional_inputs, "timestamps must be passed in for PointDecoder"
        y = self.forward_inner(xs, todo, send_to_cpu, callback, return_latents=return_latents, **kwargs)
        return y
    
    def get_matepoint_tensors(self):
        return []


def get_bbox(bbox, tensor):
    '''
    bbox is (xmin,xmax,ymin,ymax), floating point number between 0 and 1
    tensor should be shape (B,C,H,W)
    '''
    xmin,xmax,ymin,ymax = bbox
    B,C,H,W = tensor.shape
    assert xmin >= 0 and xmax <= 1 and ymin >= 0 and ymax <= 1,f"bbox out of range {bbox}"
    assert xmin < xmax and ymin < ymax, f"bbox is invalid {bbox}"
    xmin = int(xmin*W)
    xmax = int(xmax*W)
    ymin = int(ymin*H)
    ymax = int(ymax*H)
    return tensor[:,:,ymin:ymax,xmin:xmax]


if __name__ == "__main__":

    def memprint(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print("vram", n, torch.cuda.max_memory_allocated() / 1024**2)

    from train.trainer import * # <--- comment this back in if not using DDP

    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t' ]

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    timesteps = [0,6,24,48]#,36]
    
    """
    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh),
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        only_at_z=[0,3,6,9,12,15,18,21],
        requested_dates = tdates
        ))

    config.lr_sched.schedule_dts = True
    """

    from model_latlon.top import ForecastModel, ForecastModelConfig
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 512,
        pr_depth = 8,
        encdec_tr_depth = 2,
        patch_size=(4, 8, 8),
        dec_sublatent=64,
        deconv_mlp_dim=256,
        deeper_m2g=True,
        n_chunks_dec=20
    )
    from model_latlon.heal import HealLatentMesh, HealMesh, SimpleHealEncoder, GNNyHealDecoder, IcoSlide3D, SimpleHealDecoder
    hmesh = HealMesh(8, do_output=True)
    hlmesh = HealLatentMesh(depth=5, D=5, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh
    conf.hlmesh = hlmesh
    conf.update()

    enc = SimpleHealEncoder(imesh, conf, compile=False)
    proc = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
    #dec = SimpleHealDecoder(omesh, conf, compile=True)
    dec = GNNyHealDecoder(omesh, conf, compile=False)
    model = EnsembleForecastModel(conf, encoders=[enc], processors={'6': proc}, decoders=[dec])
    print(model)
    print_total_params(model)
    cuda = torch.device('cuda')
    model = model.to(cuda)
    for i in range(3):
        #if i == 1: torch.cuda.memory._record_memory_history(max_entries=100000)
        torch.cuda.synchronize()
        _t0 = time.time()
        t0 = torch.tensor([int((datetime(1997, 6, 21)-datetime(1970,1,1)).total_seconds())], dtype=torch.int64).to(cuda)
        dummy = torch.randn(1, 720, 1440, 5*16+4+len(extra_output), dtype=torch.float16).to(cuda)
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            a = model([dummy, t0], [0,6,48])
            loss = 0
            for dh in a:
                print(dh, a[dh])
                if dh == 'latent_l2': loss += a[dh] * 1e-3
                else: loss += dec.compute_loss(a[dh][0], dummy)
            #with torch.no_grad():
            #    for dh in a:
            #        dec.compute_errors(a[dh][0], dummy)
            del a
            print("fw done")
            loss.backward()
            print("bw done")
        memprint("backward")
        torch.cuda.synchronize()
        print("stuff took", time.time()-_t0)
        import gc
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
    #torch.cuda.memory._dump_snapshot("/fast/memdump_ens.pickle")

        #import pdb; pdb.set_trace()
    exit()




if __name__ == "__main__":
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    fm = ForecastModel(mesh,ForecastModelConfig(mesh)).to('cuda')
    #print(fm)
    print_total_params(fm) 

    ForecastCombinedDiffusion(fm,None)
