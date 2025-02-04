from utils import *
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_latlon.primatives2d import * 
from meshes import BalloonData, get_Dec23_meshes, MicrowaveData, SurfaceData, SatwindData, RadiosondeData
from model_latlon.data import N_ADDL_VARS, get_additional_vars
from model_latlon.cross_attn import CrossTransformer
from model_latlon.primatives2d import call_checkpointed
try:
    import torch_scatter
except:
    print("on this day we are thankful for Davy Ragland, a true innovator in software engineering")
    import os
    os.system('pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html')
    import torch_scatter

from model_latlon.convenient_transformer import ConvenientTransformer

def get_background(background, D, H, W, Cl):
    if background is None:
        return None
    elif background == 'full':
        return nn.Parameter(torch.randn(1,D,H,W,Cl))
    elif background == 'single':
        return nn.Parameter(torch.randn(Cl))
    else: assert False

class Mlp(nn.Module):
    def __init__(self, dim, out_dim, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, out_dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

from torch import Tensor
def hash_tensor(x):
    return x.detach().float().sum().item()

def get_DWHC(config):
    D = len(config.latent_levels)+1
    H = len(config.latent_lats)
    W = len(config.latent_lons)
    C = config.latent_size
    return D,H,W,C

def get_latent_flat(latent_inds,config,full_column):
    D,H,W,Cl = get_DWHC(config)
    if full_column:
        latent_space = torch.zeros([1,D,H*W,Cl], device=latent_inds.device, dtype=torch.float16)
        flat = latent_inds[:,0]*W + latent_inds[:,1]
        assert latent_inds.shape[1] == 2, "latent_inds must have 2 dimensions, for full-column"
    else:
        latent_space = torch.zeros([1,D*H*W,Cl], device=latent_inds.device, dtype=torch.float16)
        flat = latent_inds[:,0]*H*W + latent_inds[:,1]*W + latent_inds[:,2]
        assert latent_inds.shape[1] == 3, "latent_inds must have 3 dimensions, for non-full-column"
    return latent_space,flat

class ObTrEncoder(nn.Module):
    def __init__(self, mesh, config, tr_dim=None, out_dim=None):
        super(ObTrEncoder, self).__init__()
        self.out_dim = out_dim
        self.mesh = mesh
        self.config = config
        c = config
        self.full_column = 'pres_hpa' not in mesh.full_varlist
        self.num_bg_tok = self.out_dim//tr_dim * (
            len(self.config.latent_levels)+1 if self.full_column else 1) # if it doesn't have pres_hpa, then it learns over full column.
        print(f"out_dim: {self.out_dim}, num_bg_tok: {self.num_bg_tok}, tr_dim: {tr_dim}, latent_size: {c.latent_size}")
        self.tr_dim = tr_dim
        self.ob_background = nn.Parameter(torch.randn(self.num_bg_tok, tr_dim))
        self.ob_tr = ConvenientTransformer(tr_dim, depth=6, dim_head=32)
        # og run is (2,16)
        # bigger is (3,16)
        self.ob2latent = nn.Linear(self.out_dim, c.latent_size)
        #self.ob2latent = nn.Linear(c.latent_size, c.latent_size)
        
        self.direct = nn.Linear(tr_dim, tr_dim*self.num_bg_tok)

    def forward(self, x, latent_inds):
        return self.forward_vectorized(x, latent_inds)

    def forward_vectorized(self, x, latent_inds):
        N,C = x.shape
        assert C == self.tr_dim, "Input tensor must have the same number of features as the transformer"
        D,H,W,Cl = get_DWHC(self.config)
        latent_space,flat = get_latent_flat(latent_inds,self.config,self.full_column)

        if 0: x[:,0] = flat # this is useful for debugging, you can check that the scatter gets things to the correct location

        unq, counts = torch.unique(flat, return_counts=True)
        num_unique = unq.shape[0]
        c_sort = counts.sort(descending=True)

        TTPB = 1024*64 # total tokens per batch, NOTE: this may need to be configurable
        si = 0
        
        while si < num_unique:
    
            Lmax = c_sort.values[si].item()   # max seq length in this chunk
            if Lmax >= 255:
                print("Lmax is too big", Lmax, "FUCKIT im skipping")
                si += 1
                continue

            B = TTPB // (self.num_bg_tok + Lmax) 
            
            rB = min(si+B, num_unique) - si               # real batch size for this set    
            bf = unq[c_sort.indices[si:si+rB]]            # flat indexes of latent for this batch 
            assert len(torch.unique(bf)) == len(bf)

            # Create batch tensor, fill background≈
            tr_x = torch.zeros(rB, self.num_bg_tok + Lmax, self.tr_dim, 
                            device=x.device, dtype=x.dtype)
            tr_x[:, :self.num_bg_tok] = self.ob_background

            #with torch.no_grad():

            # Build a mask telling which rows in x belong to each bf
            mask = (flat[:,None] == bf[None,:])                             # [N, rB]
            off = mask.cumsum(dim=0,dtype=torch.int8) - 1                   # offset for each observation
            row_idx, col_idx = mask.nonzero(as_tuple=True)

            # Scatter the actual obs into tr_x
            tr_x[col_idx, (self.num_bg_tok + off[row_idx, col_idx]).to(torch.int)] = x[row_idx]
            
            del off; del mask; del col_idx; del row_idx

            def ch1(x_tr):
                x_tr[:,:self.num_bg_tok] += self.direct(x_tr[:,:self.num_bg_tok]).mean(dim=1).view(-1,self.num_bg_tok,self.tr_dim) 
                x_tr = self.ob_tr(x_tr)
                return x_tr

            out = call_checkpointed(ch1, tr_x, checkpoint_type=self.config.checkpoint_type)
            if not self.full_column:
                latent_space[0, bf] = self.ob2latent(out[:, :self.num_bg_tok].flatten(1))
            else:
                assert bf.min() >= 0
                assert bf.max() < latent_space.shape[2]
                latent_space[0, :,bf] = self.ob2latent(out[:, :self.num_bg_tok].reshape(out.shape[0], D, self.out_dim)).permute(1, 0, 2)
            si += rB
        latent_space = latent_space.view(1,D,H,W,Cl)
        return latent_space

class ObCrossTrEncoder(nn.Module):
    def __init__(self, mesh, config, tr_dim=None, bg_tok_total_dim=None):
        super(ObCrossTrEncoder, self).__init__()
        self.bg_tok_total_dim = bg_tok_total_dim # out_dim
        self.mesh = mesh
        self.config = config
        c = config
        self.full_column = 'pres_hpa' not in mesh.full_varlist
        self.num_bg_tok = self.bg_tok_total_dim//tr_dim * (
            len(self.config.latent_levels)+1 if self.full_column else 1) # if it doesn't have pres_hpa, then it learns over full column.
        print(f"bg_tok_total_dim: {self.bg_tok_total_dim}, num_bg_tok: {self.num_bg_tok}, tr_dim: {tr_dim}, latent_size: {c.latent_size}")
        self.tr_dim = tr_dim
        self.bg_tok_embed = nn.Parameter(torch.randn(self.num_bg_tok, tr_dim)) # ob_background
        self.ob_to_bg_transformer = CrossTransformer(tr_dim, depth=8, dim_head=32) # ob_tr
        # og run is (2,16)
        # bigger is (3,16)
        self.bg_tok_to_latent_patch = nn.Linear(self.bg_tok_total_dim, c.latent_size) # ob2latent
        #self.ob2latent = nn.Linear(c.latent_size, c.latent_size)
        
    def forward(self, x, latent_inds):
        return self.forward_vectorized(x, latent_inds)

    def forward_vectorized(self, x, latent_inds):
        N,C = x.shape
        assert C == self.tr_dim, "Input tensor must have the same number of features as the transformer"
        D,H,W,Cl = get_DWHC(self.config)
        latent_space,flat = get_latent_flat(latent_inds,self.config,self.full_column)

        if 0: x[:,0] = flat # this is useful for debugging, you can check that the scatter gets things to the correct location

        unq, counts = torch.unique(flat, return_counts=True)
        num_unique = unq.shape[0]
        c_sort = counts.sort(descending=True)

        TTPB = 1024*64 # total tokens per batch, NOTE: this may need to be configurable
        sorted_i = 0
        
        while sorted_i < num_unique:
    
            Lmax = c_sort.values[sorted_i].item()   # max seq length in this chunk
            if Lmax >= 255:
                print("Lmax is too big", Lmax, "FUCKIT im skipping")
                sorted_i += 1
                continue

            B = TTPB // (self.num_bg_tok + Lmax) 
            
            rB = min(sorted_i+B, num_unique) - sorted_i               # real batch size for this set    
            bf = unq[c_sort.indices[sorted_i:sorted_i+rB]]            # flat indexes of latent for this batch 
            assert len(torch.unique(bf)) == len(bf)

            tr_x = torch.zeros(rB, self.num_bg_tok, self.tr_dim, device=x.device, dtype=x.dtype)
            tr_x[:] = self.bg_tok_embed

            tr_ctx = torch.zeros(rB, Lmax, self.tr_dim, device=x.device, dtype=x.dtype)
            #with torch.no_grad():

            # Build a mask telling which rows in x belong to each bf
            mask = (flat[:,None] == bf[None,:])                             # [N, rB]
            off = mask.cumsum(dim=0,dtype=torch.int8) - 1                   # offset for each observation
            row_idx, col_idx = mask.nonzero(as_tuple=True)

            # Scatter the actual obs into tr_x
            tr_ctx[col_idx,off[row_idx, col_idx].to(torch.int)] = x[row_idx]        
            del off; del mask; del col_idx; del row_idx

            ctx_mask = ~(tr_ctx == 0).all(dim=-1)
            ctx_mask = torch.where(ctx_mask, 0.0, float('-inf'))

            def ch1(tr_x, tr_ctx, ctx_mask):
                tr_x = self.ob_to_bg_transformer(tr_x, ctx=tr_ctx, ctx_mask=ctx_mask)
                return tr_x

            out = call_checkpointed(ch1, tr_x, tr_ctx, ctx_mask, checkpoint_type=self.config.checkpoint_type)
            if not self.full_column:
                latent_space[0, bf] = self.bg_tok_to_latent_patch(out[:, :self.num_bg_tok].flatten(1))
            else:
                assert bf.min() >= 0
                assert bf.max() < latent_space.shape[2]
                latent_space[0, :,bf] = self.bg_tok_to_latent_patch(out[:, :self.num_bg_tok].reshape(out.shape[0], D, self.bg_tok_total_dim)).permute(1, 0, 2)
            sorted_i += rB
        latent_space = latent_space.view(1,D,H,W,Cl)
        return latent_space

def get_pos_inds_ths(x,mesh,config): # get position indices and thetas
    inds = [] ; ths = []
    res = config.latent_res
    if 'pres_hpa' in mesh.full_varlist:
        pres = x[:,mesh.full_varlist.index('pres_hpa')]
        llt = torch.tensor(config.latent_levels,device=x.device)
        levelsi = torch.searchsorted(llt, pres).to(torch.int64); levelsi = torch.clip(levelsi,0,len(llt)-1)
        inds.append(levelsi)
        llt = torch.tensor([config.outputs[0].levels[0]]+config.latent_levels,device=x.device)
        levels_th = (pres - llt[levelsi]) / (llt[levelsi+1] - llt[levelsi])
        ths.append(levels_th)


    lat,lon = x[:,mesh.full_varlist.index('lat_deg')], x[:,mesh.full_varlist.index('lon_deg')]
    lat = lat.float()
    lon = lon.float()
    lat[lat==-90] += 1e-4
    lati = ((90. - lat) // res).to(torch.int64)
    loni = (((180 // res) + (lon + 180.) // res) % (360 // res)).to(torch.int64)
    assert (lati >= 0).all()
    assert (lati < 90).all()
    inds += [lati,loni]
    lat_th = (-lat % res) / res
    lon_th = (lon % res) / res
    assert (loni >= 0).all()
    assert (loni < 180).all()
    ths += [lat_th,lon_th]
    assert 'reltime_hours' in mesh.full_varlist
    t_th = (x[:,mesh.full_varlist.index('reltime_hours')]+1)
    ths += [t_th]
    return inds, ths

def final_normalize(x,mesh):
    if 'lat_deg' in mesh.full_varlist: x[:,mesh.full_varlist.index('lat_deg')] /= 90.
    if 'lon_deg' in mesh.full_varlist: x[:,mesh.full_varlist.index('lon_deg')] /= 180.
    if 'pres_hpa' in mesh.full_varlist: x[:,mesh.full_varlist.index('pres_hpa')] /= 1000.
    if 'reltime_hours' in mesh.full_varlist: x[:,mesh.full_varlist.index('reltime_hours')] /= mesh.da_window_size/2 
    return x

class PointObPrEncoder(nn.Module):
    def __init__(self,mesh,config, tr_dim=256, out_dim=2048, combine_type="cross"):
        super(PointObPrEncoder, self).__init__()
        D,H,W,C = get_DWHC(config)
        self.full_column = 'pres_hpa' not in mesh.full_varlist
        self.mesh = mesh
        self.config = config
        self.tr_dim = tr_dim
        self.emb_dim1 = len(mesh.full_varlist) 
        self.emb_dim1 += len(mesh.indices) * (DEFAULT_POSEMB_DIM)
        self.ob_mlp = Mlp(self.emb_dim1,tr_dim)
        self.spacial_embed = nn.Linear(len(mesh.indices)*DEFAULT_POSEMB_DIM,tr_dim) 
        self.type_embedding = nn.Parameter(torch.randn(config.latent_size)) # this tells the DA transformer what type of obs this is.
        self.combine_type = combine_type
        if combine_type == "transformer":
            self.ob_tr = ObTrEncoder(mesh,config,tr_dim,out_dim)
        elif combine_type == "cross":
            self.ob_tr = ObCrossTrEncoder(mesh,config,tr_dim,out_dim)
        elif combine_type == "scattermean":
            self.to_latent = nn.Linear(tr_dim,C*D if self.full_column else C)
        elif combine_type == "old":
            pass #maybe implement old method here.

    def forward(self, x):
        #assert x.dtype == self.mesh.encoder_input_dtype, f"Input tensor must be {self.mesh.encoder_input_dtype}, but got {x.dtype}"
        N, Cob = x.shape
        assert Cob == len(self.mesh.full_varlist), "Input tensor must have the same number of features as the mesh"
        inds, ths = get_pos_inds_ths(x,self.mesh,self.config)
        latent_inds = torch.stack(inds,dim=-1)
        posemb = sincos_spacial_embed(torch.stack(ths,dim=-1))
        x = final_normalize(x,self.mesh)
        x = torch.cat([x,posemb],dim=-1).to(torch.float16)
        assert -11 < x.min() and x.max() < 11, f"very sus x value {x}"

        x = self.ob_mlp(x) + self.spacial_embed(posemb)
        if self.combine_type in ["transformer", "cross"]:
            latent_space = self.ob_tr(x,latent_inds)
        elif self.combine_type == "scattermean":
            latent_space = self.simple_scattermean(x,latent_inds)
        elif self.combine_type == "old":
            pass #maybe implement old method here.
        latent_space += self.type_embedding
        return latent_space
    
    def simple_scattermean(self,x,latent_inds):
        D, H, W, C = get_DWHC(self.config)
        full_column = 'pres_hpa' not in self.mesh.full_varlist
        if full_column:
            flat = latent_inds[:,0]*W + latent_inds[:,1]
            means = torch_scatter.scatter_mean(x, flat, dim=0, dim_size=H*W)
            means = self.to_latent(means.view(1,1,H,W,self.tr_dim)).view(1,H,W,C,D).permute(0,4,1,2,3)
        else:
            flat = latent_inds[:,0]*H*W + latent_inds[:,1]*W + latent_inds[:,2]
            means = torch_scatter.scatter_mean(x, flat, dim=0, dim_size=D*H*W)
            means = self.to_latent(means.view(1,D,H,W,self.tr_dim))
        return means

    

class PointObSfcEncoder(nn.Module):
    def __init__(self,mesh,config, tr_dim=128, tr_depth=2, tr_tokens=4, background=None):
        super(PointObSfcEncoder, self).__init__()
        assert isinstance(mesh, SurfaceData), "PointObPrEncoder is only compatible with SurfaceData"
        self.mesh = mesh
        self.config = config
        c = config
        self.emb_dim1 = len(mesh.full_varlist) 
        self.emb_dim1 += len(mesh.indices) * (1+(DEFAULT_POSEMB_NUM) * 2)# +1 is including thetas in posemb

        self.tr_dim = tr_dim
        self.tr_depth = tr_depth
        self.tr_tokens = tr_tokens
        self.token_embedding = nn.Linear(self.emb_dim1, self.tr_dim * self.tr_tokens)
        self.tr_encoder = ConvenientTransformer(depth=self.tr_depth, dim=self.tr_dim, dim_head=16)#, make_tokens=self.tr_tokens)
        self.token_combiner = nn.Linear(self.tr_dim * self.tr_tokens, self.config.latent_size)

        D, H, W, Cl = 1, len(self.config.latent_lats), len(self.config.latent_lons), self.config.latent_size
        self.background = get_background(background, D, H, W, Cl)

    def forward(self, x):
        x = rand_subset(x,dim=0,N=2**16-2)
        N, Cob = x.shape
        assert Cob == len(self.mesh.full_varlist), "Input tensor must have the same number of features as the mesh"
        assert x.dtype == torch.float32, "Input tensor must be float32"

        lat,lon = x[:,self.mesh.full_varlist.index('lat_deg')], x[:,self.mesh.full_varlist.index('lon_deg')]
        res = self.config.latent_res
        lat[lat==-90] += 1e-4 # fuck the penguins, fuck whoever came up with the idea of putting a station at the singularity

        lati = ((90. - lat) // res).to(torch.int64)
        loni = (((180 // res) + (lon + 180.) // res) % (360 // res)).to(torch.int64)
        lat_th = (-lat % res) / res
        lon_th = (lon % res) / res
        t_th = (x[:,self.mesh.full_varlist.index('reltime_hours')]+1)

        latent_inds = torch.stack([lati,loni],dim=-1)
        posemb = torch.stack([lat_th,lon_th,t_th],dim=-1)
        posemb = sin_posembed(posemb)

        x[:,self.mesh.full_varlist.index('lat_deg')] /= 90.
        x[:,self.mesh.full_varlist.index('lon_deg')] /= 180.
        x[:,self.mesh.full_varlist.index('reltime_hours')] /= self.mesh.da_window_size/2 

        x = torch.cat([x,posemb],dim=-1)
        x = x.to(torch.float16)
        assert -11 < x.min() and x.max() < 11, f"very sus x value {x}"

        x = self.token_embedding(x).view(N,self.tr_tokens,self.tr_dim)
        #print("x before: ",x)
        x = self.tr_encoder(x)
        x = x.view(N,-1)
        x = self.token_combiner(x)

        D, H, W, Cl = 1, len(self.config.latent_lats), len(self.config.latent_lons), self.config.latent_size

        flat = latent_inds[:,0]*W + latent_inds[:,1]
        assert flat.max() < D*H*W
        assert flat.min() >= 0
        reduced = torch_scatter.scatter_mean(x,flat,dim=0,dim_size=D*H*W)
        latent_space = reduced.view(1,D,H,W,Cl)
        if self.background is not None: latent_space += self.background
        #TODO: Count embedding maybe

        return latent_space

class MicrowaveEncoder(nn.Module):
    def __init__(self,mesh, config, tr_dim=128, tr_depth=1, tr_tokens=4, background=None):
        super(MicrowaveEncoder, self).__init__()
        self.mesh = mesh
        self.config = config
        self.emb_dim1 = len(mesh.full_varlist) + 3 # (lat, lon, time)

        self.tr_dim = tr_dim
        self.tr_depth = tr_depth
        self.tr_tokens = tr_tokens
        self.token_embedding = nn.Linear(self.emb_dim1, self.tr_dim * self.tr_tokens)
        if 0:
            tr_layers = nn.TransformerEncoderLayer(d_model=self.tr_dim, dim_feedforward=self.tr_dim*4,nhead=self.tr_dim//16,activation="gelu",batch_first=True,dropout=0)
            self.tr_encoder = nn.TransformerEncoder(tr_layers, num_layers=self.tr_depth)
        else:
            self.tr_encoder = ConvenientTransformer(depth=self.tr_depth, dim=self.tr_dim, dim_head=16)#, make_tokens=self.tr_tokens)
        #self.tr_encoder = nn.Identity()
        self.token_combiner = nn.Linear(self.tr_dim * self.tr_tokens, self.config.latent_size * (len(self.config.latent_levels)+1))
        self.num_obs_embed = nn.Linear(1,self.config.latent_size)

        D, H, W, Cl = len(self.config.latent_levels)+1, len(self.config.latent_lats), len(self.config.latent_lons), self.config.latent_size
        self.background = get_background(background, D, H, W, Cl)
        #self.background = nn.Parameter(torch.randn(1,D,H,W,Cl))

    def forward(self, x):
        #print("MicrowaveEncoder got x of shape: ", x.shape)
        N, Cob = x.shape
        assert Cob == len(self.mesh.full_varlist) + 2, "Input tensor must have the same number of features as the mesh (plus lat lon)"

        lat,lon = x[:,-2].float() * 180/np.pi, x[:,-1].float() * 180/np.pi
        x = x[:,:-2]
        N, Cob = x.shape
        assert Cob == len(self.mesh.full_varlist), "now forreal"
        res = self.config.latent_res

        lati = ((90. - lat) // res).to(torch.int64)
        loni = (((180 // res) + (lon + 180.) // res) % (360 // res)).to(torch.int64)
        lat_th = (-lat % res) / res
        lon_th = (lon % res) / res
        t_th = (x[:,self.mesh.full_varlist.index('reltime_hours')]/self.mesh.da_window_size + 0.5)

        latent_inds = torch.stack([lati,loni],dim=-1)
        posemb = torch.stack([lat_th,lon_th,t_th],dim=-1).to(torch.float16)
        #posemb = sin_posembed(posemb)

        x = torch.cat([posemb,x],dim=-1)
        x = x.to(torch.float16)
        del posemb, lat_th, lon_th, t_th, lati, loni

        x = self.token_embedding(x).view(N,self.tr_tokens,self.tr_dim)
        # too big to fit in memory, split into batches of size 50k:
        batch_size = 25_000
        x_batches = []

        def torch_cp(x, *args):
            return call_checkpointed(x, *args, checkpoint_type="torch")
        #self.tr_encoder.checkpointfn = torch_cp

        #dev = x.device
        #cpu = torch.device("cuda")
        for i in range(0, N, batch_size):
            # Get batch indices
            end_idx = min(i + batch_size, N)
            #print("doing batch", i, end_idx)
            # Process batch through transformer
            batch = x[i:end_idx]
            batch = self.tr_encoder(batch)
            batch = batch.view(end_idx - i, -1)
            #batch = self.token_combiner(batch)
            x_batches.append(batch)
        x = torch.cat(x_batches, dim=0)#.to(dev)
        del x_batches
        del batch
        #print("merged", x.shape)
        x = self.token_combiner(x)
        #x = torch_cp(self.token_combiner, x)
        #print("token combiner", x.shape)
        #print("done with batches", x.shape, N)
        x = x.view(N, len(self.config.latent_levels)+1, self.config.latent_size)

        """
        print("hiya", N, self.tr_tokens, self.tr_dim, x.shape)
        x = self.tr_encoder(x)
        x = x.view(N,-1)
        x = self.token_combiner(x)
        """

        D, H, W, Cl = len(self.config.latent_levels)+1, len(self.config.latent_lats), len(self.config.latent_lons), self.config.latent_size

        #latent_space2 = torch.zeros(1,D,H,W,Cl,device=x.device,dtype=torch.float16)
        #torch.cuda.synchronize()
        #aa = time.time()

        #uinds, inverse_indices, counts = torch.unique(latent_inds, dim=0, return_inverse=True, return_counts=True)
        #print("uh", counts.shape)
        #print("counts", counts.max())

        if 1:
            #import pdb; pdb.set_trace()
            flat = latent_inds[:,0] * W + latent_inds[:,1]
            del latent_inds
            means = torch_scatter.scatter_mean(x, flat, dim=0, dim_size=H*W)
            counts = torch_scatter.scatter_add(
                torch.ones_like(flat, dtype=x.dtype, device=x.device), 
                flat, 
                dim=0, 
                dim_size=H*W
            ) # TODO: inefficient. should be able to do use torch.unique maybe?
            numobs = self.num_obs_embed(torch.log(1+counts.unsqueeze(1)).unsqueeze(1) )
            numobs[counts == 0] = 0 # TODO: learned 0?
            means += numobs
            latent_space2 = means.view(1, H, W, D, Cl).permute(0,3,1,2,4)
            assert latent_space2.dtype == torch.float16

            #torch.cuda.synchronize()
            if self.background is not None: latent_space2 += self.background
            #print("neo took", time.time()-aa)
            latent_space = latent_space2

        else:
            # OLD: slow
            latent_space = torch.zeros(1,D,H,W,Cl,device=x.device,dtype=torch.float16)

            #torch.cuda.synchronize()
            #aa = time.time()
            uinds = torch.unique(latent_inds,dim=0)
            for i in range(uinds.shape[0]):
                inds = uinds[i,:]
                to_sum = torch.where(torch.all(latent_inds==inds,dim=1))[0]
                n = to_sum.shape[0]
                xp = torch.mean(x[to_sum,:],dim=0)
                xp += self.num_obs_embed(torch.tensor([n],device=x.device,dtype=x.dtype))
                latent_space[0,:,inds[0],inds[1],:] = xp

            #torch.cuda.synchronize()
            #print("aaa python took", time.time()-aa)
            #print("err is", torch.sqrt(torch.mean(torch.square(latent_space.float()-latent_space2.float()))))

        #assert torch.all(latent_space[:,-1,:,:,:] == 0), "Last depth (sfc) must be zero, this is a pr encoder"

        return latent_space


        # x looks like this
        # - dt (always negative, -1 to 0 [hours])
        # - satellite id / said. should just convert to one hot of size len(mesh.sats) / do an embedding
        # - lat, lon: should do whatever embedding we do for point obs on the latent space presumably
        # - saza/soza: some relevant angles. should convert to sine and cosine and treat it as part of the observation
        # - self.nchan channels. occasionally some nans that should be converted to 0 and masked off with the _present? variables
        pass


def measure_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
    return 0

def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

def memory_test(model, input_tensor, clear_cache=True):
    if clear_cache:
        reset_peak_memory()
    
    # Forward pass
    torch.cuda.synchronize()
    t0 = time.time()
    output = model(input_tensor)
    forward_peak = measure_peak_memory()
    torch.cuda.synchronize()
    dtf = time.time() - t0
    
    # Backward pass
    loss = output.mean()
    if clear_cache:
        reset_peak_memory()
    
    torch.cuda.synchronize()
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    dtb = time.time() - t0
    backward_peak = measure_peak_memory()
    
    return {
        'forward_peak_mb': forward_peak,
        'backward_peak_mb': backward_peak,
        'dtf': dtf*1000,
        'dbw': dtb*1000,
        'output_shape': tuple(output.shape)
    }


if __name__ == "__main__" and os.environ.get('JOAN'):
    from model_latlon.top import ForecastModelConfig
    from datasets import MicrowaveDataset
    from model_latlon.da_transformer import DATransformer

    device = 'cuda'

    atms_mesh = MicrowaveData("1bamua") # or 1bamua
    #print(atms_mesh.full_varlist)
    loader = MicrowaveDataset(atms_mesh)
    x = loader.load_data(1655812800)
    #import pdb; pdb.set_trace()
    #plot_igra_data_tensor(x,atms_mesh)
    x = x.to(device)
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    
    config = ForecastModelConfig(
        None,
        outputs=[omesh],
        #checkpoint_type="torch",
        encdec_tr_depth=4,
        latent_size=896,
        window_size=(3,5,7),
        weight_eps=0.01,
    )

    #config = ForecastModelConfig([get_Dec23_meshes()[0],atms_mesh])
    atms_encoder = MicrowaveEncoder(atms_mesh,config).to(device)
    #tr = DATransformer(config).to(device)
    print_total_params(atms_encoder)
    x = x[:150000]
    print(x.shape)
    with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        for i in range(5):
            print(i, memory_test(atms_encoder, x.clone()))
        exit()
        t0 = time.time()
        lx = atms_encoder(x)
        print("time", time.time()-t0)
        print("second time")
        lx = atms_encoder(x.clone())
        #lx = tr([lx])


    exit()

def test_balloon():
    from model_latlon.config import ForecastModelConfig
    from datasets import IgraDataset
    igra_mesh = BalloonData()
    loader = IgraDataset(igra_mesh)
    x = loader.load_data(1655812800)
    x = x.to('cuda')
    conf = ForecastModelConfig(inputs=[get_joansucks_omesh()])
    model = PointObPrEncoder(igra_mesh, conf).to('cuda')
    print(x.shape)
    with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        for i in range(5):
            y = model(x)
            print(y.shape)


if __name__ == "__main__":
    from utils import *
    test_balloon()


