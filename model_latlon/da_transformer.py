from utils import *
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_latlon.primatives2d import print_total_params, rand_subset
from meshes import BalloonData, get_Dec23_meshes
from model_latlon.config import ForecastModelConfig
from model_latlon.transformer3d import Natten3DTransformerBlock, tr_pad, tr_unpad, posemb_sincos_3d
from model_latlon.primatives2d import call_checkpointed, matepoint
from model_latlon.top import SlideLayers3D
from model_latlon.encoder import SimpleConvEncoder

from model_latlon.convenient_transformer import ConvenientTransformer
from model_latlon.process import *


class DATransformer(nn.Module):
    def __init__(self,config : ForecastModelConfig):
        super().__init__()
        self.config = config

        self.depth = config.da_depth
        self.dim = self.config.latent_size
        self.num_heads = self.config.latent_size//self.config.dims_per_head
        self.window_size = self.config.window_size

        tr3ds = []
        tr2ds = []
        for _ in range(self.depth):            
            #tr = nn.TransformerEncoderLayer(d_model=self.dim, dim_feedforward=self.dim*4,
            #                            nhead=self.num_heads,activation="gelu", dropout=0, norm_first=True) # NOTE: Batch dim is NOT first here, because it allows us to avoid permutes
            tr = ConvenientTransformer(depth=1, dim=self.dim, dim_head=self.dim//self.num_heads, batch_first=False)#, make_tokens=self.tr_tokens)
            tr2ds.append(tr)
            L = len(self.config.latent_levels)+1
            if L % 2 == 0:
                L -= 1 # rip models with 28lev
            tr = Natten3DTransformerBlock(
                self.dim,
                self.num_heads,
                window_size=(L,3,3),
                embedding_module="nothing"
            )
            tr3ds.append(tr)

        self.tr3ds = nn.ModuleList(tr3ds)
        self.tr2ds = nn.ModuleList(tr2ds)

        
    def forward(self, latents):
        N = len(latents)
        B,D,H,W,C = latents[0].shape
        assert all(l.shape == (B,D,H,W,C) for l in latents), f"All latents must have the same shape. {latents[0].shape}"
        assert B == 1
        #assert N == 2

        x = torch.cat(latents,dim=0)
        if self.config.da_perturber != 0: anl_save = x[0:1].clone()
        pe = posemb_sincos_3d(((0, D, H, W, C), x.device, x.dtype), temperature=1000).reshape(1, D,H,W,C)
        x += pe
        for i in range(self.depth):
            x = x.reshape(N*D,H*W,C)

            B = 4
            if B > 1:
                seq_len = H*W
                batch_size = seq_len // B
                assert seq_len % B == 0
                
                outputs = []
                for j in range(0, seq_len, batch_size):
                    batch = x[:, j:j+batch_size, :].clone()
                    outputs.append(call_checkpointed(self.tr2ds[i], batch, checkpoint_type=self.config.checkpoint_type))
                x = torch.cat(outputs, dim=1)
                del outputs
            else:
                x = call_checkpointed(self.tr2ds[i],x, checkpoint_type=self.config.checkpoint_type)

            x = x.reshape(N,D,H,W,C)

            def ch1(x):
                x = tr_unpad(self.tr3ds[i](tr_pad(x.clone(), self.config.window_size)), self.config.window_size)
                return x 

            x[0:1] = call_checkpointed(ch1,x[0:1].clone(), checkpoint_type=self.config.checkpoint_type)

            #x[0:1] = tr_unpad(call_checkpointed(self.tr3ds[i],tr_pad(x[0:1].clone(), self.config.window_size)), self.config.window_size) # NATEN transformer only looks at the first element which is the latent that we are assimilating into
        if self.config.da_perturber != 0:
            print("perturbing", self.config.da_perturber)
            return anl_save + self.config.da_perturber * x[0:1]
        else: return x[0:1]


class DAForecastModel(nn.Module, SourceCodeLogger):
    def __init__(self, forecast_model, obs_encoders, da_transformer):
        super().__init__()
        self.forecast_model = forecast_model
        self.da_transformer = da_transformer
        self.encoders = forecast_model.encoders + obs_encoders
        #self.obs_encoders = obs_encoders
        self.decoders = forecast_model.decoders
        self.config = forecast_model.config
    
    def forward(self, xs, todo):
        with torch.no_grad():
            x_6 = self.forecast_model(xs[:2] + [xs[-1]], [6], return_latents=True)[6]
        todo = [t-6 for t in todo]
        obs_encoders = self.encoders[2:]
        assert len(xs)-1-2 == len(obs_encoders)
        assert all([x.shape[0] == 1 for x in xs[:-1]])

        for i in range(len(obs_encoders)):
            N = xs[i+2][0].shape[0]
            N_subset = 100_000
            #N_subset = N
            N_subset = min(N_subset, N)
            if N_subset != N:
                # get a random subset so it fits in ram
                perm = torch.randperm(N)
                xs[i+2] = xs[i+2][:,perm][:, :N_subset]

        ob_latents = [call_checkpointed(enc,xs[i+2][0], checkpoint_type=self.config.checkpoint_type) for i,enc in enumerate(obs_encoders)]
        #ob_latents = [x_6.clone()]
        x_da = self.da_transformer([x_6] + ob_latents)
        #def tr_pad_prime(a, b):
        #    print("I'm a bigger piece of shit! I got a timeout too")
        #    time.sleep(666e-4)
        #    return tr_pad(a, b)
        x_da = call_checkpointed(tr_pad,x_da, self.config.window_size, checkpoint_type=self.config.checkpoint_type)
        #x_da = tr_pad(torch.mean(torch.cat([x_6] + ob_latents, dim=0), axis=0)[None], self.config.window_size)
        #import pdb; pdb.set_trace()
        ys = self.forecast_model([x_da, xs[-1] + 3600 * 6], todo, from_latent=True)
        return ys
    
    def get_matepoint_tensors(self):
        return self.forecast_model.get_matepoint_tensors()
    

def combine_latents(lats, encs):
    assert len(lats) == len(encs)
    types = [x.__class__.__name__ for x in encs]
    try:
        sfcidx = types.index("PointObSfcEncoder")
        assert types.count("PointObSfcEncoder") == 1
        pridx = types.index("PointObPrEncoder")
        assert lats[pridx][:, -1].shape == lats[sfcidx][:,0].shape
        lats[pridx][:, -1] += lats[sfcidx][:, 0]
        lats.pop(sfcidx)
        return lats
    except:
        return lats

class DAOnlyModel(nn.Module, SourceCodeLogger):
    def __init__(self, config, anl_encoders = [], obs_encoders = [], da_transformer = None, decoders = []):
        super().__init__()
        self.config = config
        self.anl_encoders = nn.ModuleList(anl_encoders)
        self.obs_encoders = nn.ModuleList(obs_encoders)
        self.encoders = anl_encoders + obs_encoders
        self.da_transformer = da_transformer
        self.processor = SlideLayers3D(dim=self.config.latent_size, depth=config.pr_depth[0], num_heads=config.num_heads, window_size=self.config.window_size, embedding_module=self.config.embedding_module, checkpoint_type=self.config.checkpoint_type)
        self.decoders = nn.ModuleList(decoders)
        D = len(self.config.latent_levels) + 1
        H = len(self.config.latent_lats)
        W = len(self.config.latent_lons)
        Cl = self.config.latent_size
        self.background = nn.Parameter(torch.randn(1,D,H,W,Cl))

    def forward(self, xs, todo, **kwargs):
        assert len(xs)-1 == len(self.encoders), f"len(xs)-1 must equal len(self.encoders), {len(xs)-1} != {len(self.encoders)}"
        ob_latents = []

        print("x shapes", [a.shape for a in xs[:-1]])

        for i,x in enumerate(xs[len(self.anl_encoders):-1]):
            #ob_latent = call_checkpointed(self.obs_encoders[i],random_subset(x[0]), checkpoint_type=self.config.checkpoint_type)
            with Timer(f"ob_enc {self.obs_encoders[i].mesh.string_id}",print=True):
                ob_latent = self.obs_encoders[i](x[0])
            ob_latents.append(ob_latent)
        
        ob_latents = combine_latents(ob_latents, self.obs_encoders)
        
        for o in ob_latents:
            assert o.shape == ob_latents[0].shape, f"All ob_latents must have the same shape, {o.shape} != {ob_latents[0].shape}"
        
        #anl = torch.zeros_like(ob_latents[0])
        anl = self.background

        x_da = self.da_transformer([anl] + ob_latents)
        x_da = call_checkpointed(tr_pad,x_da, self.config.window_size, checkpoint_type=self.config.checkpoint_type)
        x_da = self.processor(x_da)
        y = self.decoders[0](x_da)
        return {0:[y]}
    

class DAForecastModel2(nn.Module, SourceCodeLogger):
    def __init__(self, config, anl_encoders = [], obs_encoders = [], da_transformer = None, decoders = [], processors = None):
        super().__init__()
        self.config = config
        assert self.config.processor_dts == [6]
        self.anl_encoders = nn.ModuleList(anl_encoders)
        self.obs_encoders = nn.ModuleList(obs_encoders)
        self.encoders = anl_encoders + obs_encoders
        self.da_transformer = da_transformer
        if processors is None:
            self.processors = nn.ModuleList([SlideLayers3D(dim=self.config.latent_size, depth=config.pr_depth[0], num_heads=config.num_heads, window_size=self.config.window_size, embedding_module=self.config.embedding_module, checkpoint_type=self.config.checkpoint_type)])
        else:
            self.processors = nn.ModuleDict(processors)
        self.decoders = nn.ModuleList(decoders)
        self.op_map = make_default_op_map(self)

    def forcast_only(self,xs,todo):
        xs = [xs[0], xs[-1]]
        todo = simple_gen_todo(todo,self.config.processor_dts)
        targets = make_targets(todo,xs)
        return process(targets,self.op_map)

    def forward(self, xs, todo, **kwargs):
        #print("len of crap is", len(matepoint.Gmatepoint_ctx))
        #if len(matepoint.Gmatepoint_ctx) > 0: print(matepoint.Gmatepoint_ctx[0], matepoint.Gmatepoint_ctx[-1])
        #matepoint.Gmatepoint_ctx = []
        #print("now len of crap is", len(matepoint.Gmatepoint_ctx))
        assert len(matepoint.Gmatepoint_ctx) == 0
        assert len(xs)-1 == len(self.encoders), f"len(xs)-1 must equal len(self.encoders), {len(xs)-1} != {len(self.encoders)}"
        t0 = xs[-1]
        #print("x shapes", [a.shape for a in xs[:-1]])
        
        # encode analysis

        with torch.no_grad():
            anls =  xs[:len(self.anl_encoders)]
            anl_latents = []
            for i,x in enumerate(anls):
                dh = self.anl_encoders[i].mesh.hour_offset
                anl_latent = self.anl_encoders[i](x,t0-dh*3600)
                anl_latents.append(anl_latent)

            # process forward analysis
            for i,anl in enumerate(anl_latents):
                assert self.anl_encoders[i].mesh.hour_offset == 6
                anl_6 = self.processors['6'](anl)
                anls[i] = tr_unpad(anl_6, self.config.window_size)

        # encode obs
        obs = xs[len(self.anl_encoders):-1]
        ob_latents = []
        for i,x in enumerate(obs):
            if x.numel() == 0:
                print(f"No data available for {self.obs_encoders[i].mesh.string_id}, skipping")

                continue # no data available for this time
            x = rand_subset(x[0],dim=0,N=self.obs_encoders[i].mesh.encoder_subset).clone()
            ob_latent = call_checkpointed(self.obs_encoders[i],x, checkpoint_type=self.config.checkpoint_type)
            ob_latents.append(ob_latent)
        
        for o in ob_latents:
            assert o.shape == ob_latents[0].shape, f"All ob_latents must have the same shape, {o.shape} != {ob_latents[0].shape}"
        
        # DA
        x_da = self.da_transformer(anls + ob_latents)
        x_da = call_checkpointed(tr_pad,x_da, self.config.window_size)

        # hardcoded for now
        assert todo == [0,24]  or todo == [0]
        #x = self.processors[0](x_da)
        x = x_da
        y_0 = self.decoders[0](x)
        dic = {0: [y_0]}
        if 24 in todo:
            for i in range(4):
                x = self.processors['6'](x)
            y_24 = self.decoders[0](x)
            dic[24] = [y_24]

        return dic
