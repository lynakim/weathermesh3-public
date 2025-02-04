import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
try: # only joan needs these
    import healpy as hp
    from scipy.spatial import Delaunay
    from scipy.spatial import cKDTree
except:
    pass

from utils import *
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

#torch.manual_seed(0)

def memprint(n):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print("vram", n, torch.cuda.max_memory_allocated() / 1024**2)

def make_hgrid(nu, D, KL, KD, B):
    path = "/fast/consts/meshes/hgrid_nu%d_D%d_KL%d_KD%d_B%d.pickle" % (nu, D, KL, KD, B)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    ok = input("""

HELLO HELLO HELLO
               IT'S ME AGAIN

               ARE YOU JOAN  ? ? ?? !


               IF YOU ARE NOT JOAN. CONSIDER YOURSELF LUCKY
               AND PRAY THAT YOU DON'T EVER BECOME him


    THIS IS NOT A PLACE OF HONOR
               
        NO HIGHLY ESTEEMED CODE LIVES HERE

               YOU REALLY DON'T WANT TO BE RUNNING THIS CODE

               IF YOU'RE SEEING THIS SOMETHING WENT TERRIBLY WRONG
            
                    TAKE SHELTER
                    TAKE CARE OF YOUR LOVED ONES

                AND MESSAGE JOAN TO HANDLE THIS
               
               IF YOU ARE JOAN AND ARE READY TYPE "yes"
               
               """)
    assert ok == "yes"
    #vertices, faces = icosphere(nu)
    NPIX = hp.nside2npix(nu)
    pts = hp.pix2vec(nside=nu, ipix=list(range(NPIX)), nest=True)
    ang = hp.pix2ang(nside=nu, ipix=list(range(NPIX)), nest=True, lonlat=True)
    lonlat = torch.from_numpy(np.array(ang).T)
    vertices = np.array(pts).T
    #indices = reorder_sphere_points(vertices)
    #vertices = vertices[indices]
    idxs = np.array(list(range(vertices.shape[0])), dtype=np.int32)
    neighbors = []
    for i, v in tqdm(list(enumerate(vertices))):
        dot_products = np.sum(v * vertices, axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles = np.arccos(dot_products)
        dist = angles * 6371
        srt = np.argsort(dist)[:KL]
        neighbors.append(idxs[srt])
    neighbors = np.array(neighbors, dtype=np.int32)
    #with open('cache%d.pickle'%K, 'wb') as f:
    #    pickle.dump(neighbors, f)

    kernel_size = (KL, KD)
    Nvert = vertices.shape[0]
    K = np.prod(kernel_size)

    full2 = np.zeros((vertices.shape[0], D, np.prod(kernel_size), 2), dtype=np.int32) - 1
    full = np.zeros((vertices.shape[0], D, np.prod(kernel_size)), dtype=np.int32) - 1
    for i, neighs in tqdm(list(enumerate(neighbors))):
        for A in range(D):
            idx = 0
            for neigh in neighs:
                for a in range(-(kernel_size[1]//2), kernel_size[1]//2+1):
                    aa = A + a
                    if A < kernel_size[1]//2:  # If we're too close to start
                        aa = a + kernel_size[1]//2  # Offset from start
                    elif A >= D - kernel_size[1]//2:  # If we're too close to end
                        aa = D - kernel_size[1] + a + kernel_size[1]//2  # Offset from (D - kernel_size)
                    full[i, A, idx] = D * neigh + aa#[neigh, aa]
                    full2[i, A, idx] = [neigh, aa]
                    idx += 1
    assert (full>=0).all()
    T = Nvert*D
    #assert Nvert % B == 0
    NB = T//B
    if T % B != 0: NB += 1
    full.shape = (T, K)
    rat = []
    which = []
    for i in range(NB):
        batch = full[i*B:(i+1)*B]
        uniq = sorted(list(set(batch.flatten())))
        which.append(uniq)
        rat.append(len(uniq))
    mx = max(rat)
    which = [x + [2**32-1]*(mx-len(x)) for x in which]
    which = np.array(which, dtype=np.uint32)
    masklen = int(np.ceil(mx/32))
    mask = np.zeros((T, masklen), dtype=np.uint32)
    mask2 = np.zeros((T, mx), dtype=bool)

    full.shape = (T, K)

    for i in range(T):
            blk = i//B
            # this is an annoying little endian kinda shit but let's me do >>1 every time
            stuff = set(list(full[i]))
            assert len(stuff) == K
            idx = 0
            nb = 0
            msk = 0
            hm = 0
            for k in range(mx):
                if which[blk, k] in stuff:
                    msk |= (1 << nb)
                    mask2[i, k] = True
                    hm += 1
                nb += 1
                if nb == 32:
                    nb = 0
                    mask[i, idx] = msk
                    msk = 0
                    idx += 1
            if nb != 0: mask[i, idx] = msk
            popcount = sum([x.bit_count() for x in mask[i]])
            assert popcount == K
    
    print(neighbors.shape, which.shape, mask.shape)
    print(which.nbytes/1e6, mask.nbytes/1e6, mask2.nbytes/1e6)
    dic = {'vertices': vertices, 'which': which, 'mask': mask, 'full2': full2, 'mask2': mask2, 'lonlat': lonlat}
    pprint({k: v.shape for k,v in dic.items()})

    with open(path, 'wb') as f:
        pickle.dump(dic, f)

    return dic

def precompute_bilinear_indices_and_weights(target_lats, target_lons):
    """
    Precompute bilinear interpolation for a 0.25° global grid:
      lat = 90 down to -90 (720 points)
      lon = 0..360 (modded) (1440 points)

    Inputs:
      target_lats, target_lons: (N,) arrays in degrees.

    Returns:
      idx:     (N, 4) int array with flattened corner indices
      weights: (N, 4) float array with bilinear weights
               (each row sums to 1)
    """
    # 1) Convert lat to [0..720), clamp if out of range
    #    The "index" formula for lat is  i_f = (90 - lat) / 0.25
    lat_f = (90.0 - target_lats) / 0.25
    lat_f = np.clip(lat_f, 0, 719.9999)  # clamp within 0..~720
    i0 = np.floor(lat_f).astype(int)
    i1 = np.minimum(i0 + 1, 719)
    alpha_lat = lat_f - i0  # in [0..1], may overshoot slightly if out of range
    alpha_lat = np.clip(alpha_lat, 0, 1)

    # 2) Convert lon to [0..1440) by mod 1440
    #    The "index" formula for lon is j_f = lon / 0.25
    lon_f = target_lons / 0.25
    lon_f_mod = np.mod(lon_f, 1440)   # wrap in [0..1440)
    j0 = np.floor(lon_f_mod).astype(int)
    j1 = (j0 + 1) % 1440
    alpha_lon = lon_f_mod - j0  # in [0..1)

    # 3) Flattened corner indices:  corner = i * 1440 + j
    corner0 = i0 * 1440 + j0
    corner1 = i0 * 1440 + j1
    corner2 = i1 * 1440 + j0
    corner3 = i1 * 1440 + j1
    idx = np.column_stack([corner0, corner1, corner2, corner3])

    # 4) Bilinear weights
    w0 = (1 - alpha_lat) * (1 - alpha_lon)
    w1 = (1 - alpha_lat) * alpha_lon
    w2 = alpha_lat * (1 - alpha_lon)
    w3 = alpha_lat * alpha_lon
    weights = np.column_stack([w0, w1, w2, w3])

    return idx.astype(np.int32), weights.astype(np.float32)

def get_nearest_from_mesh(vertices, tgtlat, tgtlon, nn=4):
    tree = cKDTree(vertices)
    lon_grid, lat_grid = np.meshgrid(tgtlon, tgtlat)
    print(lon_grid.shape)
    grid_vectors = np.column_stack([
        np.cos(np.radians(lat_grid.ravel())) * np.cos(np.radians(lon_grid.ravel())),
        np.cos(np.radians(lat_grid.ravel())) * np.sin(np.radians(lon_grid.ravel())),
        np.sin(np.radians(lat_grid.ravel()))
    ])

    # 4. Find four nearest neighbors for each grid point
    distances, indices = tree.query(grid_vectors, k=nn)
    return indices.reshape(720, 1440, nn).astype(np.int32)

import iconatten

class HealLatentMesh:
    def __init__(self, depth, D, KL, KD, dim, B=8, BSh=8):
        self.NSIDE = 2**depth
        self.D = D
        self.KL = KL
        self.KD = KD
        self.B = B
        print("depth", depth, "D", D, "KL", KL, "KD", KD, "dim", dim, "B", B, "BSh", BSh)
        self.hgrid = make_hgrid(self.NSIDE, self.D, self.KL, self.KD, self.B)
        self.vertices = torch.from_numpy(self.hgrid['vertices'])
        self.Nvert = self.vertices.shape[0]
        self.which = torch.from_numpy(self.hgrid['which'])
        self.mask = torch.from_numpy(self.hgrid['mask'])
        self.BS = B
        self.BSh = BSh
        self.posemb = self.posemb_mesh(self.hgrid['lonlat'], D, dim).half()

    def posemb_mesh(self, lonlat, D, dim, T1=300, T2=10, dtype=torch.float32):
        fourier_dim = dim // 8
        omegas = torch.arange(fourier_dim) / (fourier_dim - 1)
        omega = 1. / (T1 ** omegas)
        omega2 = 1. / (T2 ** omegas)

        # Convert to 3D coordinates on unit sphere
        lon = torch.deg2rad(lonlat[:, 0])
        lat = torch.deg2rad(lonlat[:, 1])
        
        # Convert to xyz coordinates (this preserves actual spatial relationships)
        FF = 40
        x = (torch.cos(lat) * torch.cos(lon))*FF
        y = (torch.cos(lat) * torch.sin(lon))*FF
        z_coord = (torch.sin(lat))*FF
        
        def rep(xx):
            return xx.unsqueeze(1).repeat(1, D, 1)
        
        # Encode each spatial dimension
        x_emb = rep(x[:, None] * omega[None, :])
        y_emb = rep(y[:, None] * omega[None, :])
        z_emb = rep(z_coord[:, None] * omega[None, :])
        
        # Handle vertical levels
        levels = torch.arange(D)
        level_emb = levels[:, None] * omega2[None, :]
        level_emb = level_emb[None, :, :].expand(lonlat.shape[0], -1, -1)
        
        pe = torch.cat((
            torch.sin(x_emb), torch.cos(x_emb),
            torch.sin(y_emb), torch.cos(y_emb),
            torch.sin(z_emb), torch.cos(z_emb),
            torch.sin(level_emb), torch.cos(level_emb)
        ), dim=2)
        
        pe = F.pad(pe, (0, dim - (fourier_dim * 8)))
        return pe.type(dtype)

    
    def to(self, *args, **kwargs):
        self.vertices = self.vertices.to(*args, **kwargs)
        self.which = self.which.to(*args, **kwargs)
        self.mask = self.mask.to(*args, **kwargs)
        self.posemb = self.posemb.to(*args, **kwargs)
        return self

class HealMesh:
    def to(self, *args, **kwargs):
        self.g2m_wts = self.g2m_wts.to(*args, **kwargs)
        self.g2m_idx = self.g2m_idx.to(*args, **kwargs)
        if self.m2g_idx is not None: self.m2g_idx = self.m2g_idx.to(*args, **kwargs)
        return self

    def __init__(self, depth, do_output=True):
        self.NSIDE = 2**depth

        cache_path = "/fast/consts/meshes/healpix_%d.pickle" % depth
        if os.path.exists(cache_path):
            self.vertices, (self.g2m_idx, self.g2m_wts), self.m2g_idx = pickle.load(open(cache_path, "rb"))
            self.g2m_wts = torch.from_numpy(self.g2m_wts).half()
            self.g2m_idx = torch.from_numpy(self.g2m_idx)
            if do_output: self.m2g_idx = torch.from_numpy(self.m2g_idx)
            else: self.m2g_idx = None
        else:
            ok = input("""
HELLO HELLO HELLO

                    ARE you JOAN ? ! ? !

            This code is known to cause cancer in the state of California
                    it's gotten a bit better
                    but the preferred mode of operation is to still let Joan go to the sewers if anything needs to change


                you are NOT allowed to say yes if you're not joan
                    trust me it's for your own good

            SENDING THIS MESSAGE WAS IMPORTANT TO HIM. HE CONSIDERED HIMSELF TO BE A POWERFUL CODER
                       WHAT IS HERE WAS DANGEROUS AND REPULSIVE TO HIM. THIS MESSAGE IS A WARNING ABOUT DANGER


                    THE DANGER INCREASES TOWARDS THE BOTTOM OF THE FILE... AND BELOW US

                if you encountered THIS, something has gone wrong
                    please let joan know

            ARE YOU JOAN ? ! ? !
                    
                    answer "yes" or "no"
                    
                    """)
            assert ok == "yes"
            t0 = time.time()
            NPIX = hp.nside2npix(self.NSIDE)
            pts = hp.pix2vec(nside=self.NSIDE, ipix=list(range(NPIX)), nest=True)
            ang = hp.pix2ang(nside=self.NSIDE, ipix=list(range(NPIX)), nest=True, lonlat=True)
            self.vertices = np.array(pts).T
            ang = np.array(ang).T
            lonlat = ang.copy()
            lon = ang[:, 0]
            lon[lon<0] += 360
            ang[:,0] = lon
            lat = ang[:, 1]
            #self.interp_grid = [(2.0*(lon/359.75) - 1.0, 2.0 * ((90.0 - lat) / 179.75) - 1.0 ) for lat, lon in pts]
            tgtlon = np.arange(0, 360, 0.25)
            tgtlat = np.arange(90, -89.99, -0.25)
            self.ang = np.fliplr(ang)
            self.g2m = precompute_bilinear_indices_and_weights(lat, lon)
            self.m2g_indices = get_nearest_from_mesh(self.vertices, tgtlat, tgtlon)
            with open(cache_path, "wb") as f:
                pickle.dump((self.vertices, self.g2m, self.m2g_indices), f)
            print("everything", (time.time()-t0)*1000)


class CUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, a, b, BS, BSh):
        # Save inputs for backward pass
        out_L = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), dtype=torch.float32, device=q.device)
        out_M = torch.zeros_like(out_L)
        fn_name = "iconatten.forward_%d" % q.shape[3]
        out = eval(fn_name)(q, k, v, a, b, out_L, out_M, BS, BSh)
        ctx.fn_name = fn_name
        ctx.batch = (BS, BSh)
        ctx.save_for_backward(q, k, v, a, b, out, out_L, out_M)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, a, b, out, out_L, out_M = ctx.saved_tensors
        fn_name = ctx.fn_name
        BS, BSh = ctx.batch
        if BS == 8:
            fn_name += "_8"
        assert grad_output.is_contiguous()
        # Call your CUDA backward implementation
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        eval(fn_name.replace("forward", "backward"))(
            grad_q, grad_k, grad_v,
            grad_output, q, k, v, out, a, b, out_L, out_M, BS, BSh
        )
        return grad_q, grad_k, grad_v, None, None, None, None, None

class IcoAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        hlmesh = None
    ):
        super().__init__()

        assert hlmesh is not None

        self.Nvert = hlmesh.Nvert
        self.D = hlmesh.D

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.hlmesh = hlmesh

    
    def forward(self, x):
        #x = x[0]
        B = x.shape[0]
        assert x.shape[1] == self.Nvert
        assert x.shape[2] == self.D
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, self.Nvert, self.D, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, :, 0]
        q /= self.scale
        k = qkv[:, :, :, 1]
        v = qkv[:, :, :, 2]

        y = []
        for i in range(B):
            y.append(CUDAFunction.apply(q[i], k[i], v[i], self.hlmesh.which, self.hlmesh.mask, self.hlmesh.BS, self.hlmesh.BSh))
        y = torch.stack(y)
        assert y.shape[0] == B
        return self.proj(y)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    

class IcoBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        hlmesh = None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = norm_layer(dim)        
        self.attn = IcoAttention(
            dim,
            num_heads,
            qkv_bias = qkv_bias,
            hlmesh = hlmesh
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):

        B, Nvert, D, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x
    


class IcoSlide3D(nn.Module):
    def __init__(self, dim, depth, num_heads, hlmesh, checkpoint_type="matepoint"):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.checkpoint_type = checkpoint_type

        mlist = []
        for _ in range(depth):
            tb_block = IcoBlock(
                dim,
                num_heads,
                hlmesh=hlmesh
            )
            mlist.append(tb_block)
        self.blocks = nn.ModuleList(mlist)

    def forward(self, x):

        for _, blk in enumerate(self.blocks):
            x = call_checkpointed(blk, x, checkpoint_type=self.checkpoint_type)

        return x

def mccompileface(mod):
    return torch.compile(mod,
                mode="reduce-overhead",  # Most aggressive optimization mode
                fullgraph=True,      # Enable full graph optimization
                dynamic=True,       # Disable dynamic shapes for more optimization
                backend="inductor"   # Use the most optimized backend)
            )


class SimpleHealEncoder(nn.Module):

    def __init__(self, mesh, config, compile=False):
        super(SimpleHealEncoder, self).__init__()
        self.mesh = mesh
        self.config = config

        self.outer = InnerSimpleHealEncoder(mesh, config)
        if compile:
            self.outer = mccompileface(self.outer)

        c = config
        self.tr = IcoSlide3D(depth=self.config.encdec_tr_depth, num_heads=c.num_heads, dim=c.latent_size, hlmesh=self.config.hlmesh, checkpoint_type=self.config.checkpoint_type)
    
    def forward(self, x, t0s):
        assert not torch.isnan(x).any() # re-add the nan stuff
        #if torch.isnan(x_sfc).any():
        #    x_sfc = torch.nan_to_num(x_sfc, nan=0.0)
        
        addl = get_additional_vars(t0s)

        return self.tr(self.outer(x, addl))


class InnerSimpleHealEncoder(nn.Module):

    def __init__(self, mesh, config):
        super(InnerSimpleHealEncoder, self).__init__()
        self.mesh = mesh
        self.config = config
        c = config
        data,cvars = get_constant_vars(mesh)
        data = data.unsqueeze(0)
        self.register_buffer('const_data', data)
        self.total_sfc = self.mesh.n_sfc_vars + N_ADDL_VARS + len(cvars)

        patch = self.config.patch_size
        assert patch[1] == patch[2]
        self.conv_sfc = nn.Linear(self.total_sfc * patch[1] * patch[1], c.latent_size)
        self.conv_pr = nn.Linear(self.mesh.n_pr_vars * patch[0] * patch[1] * patch[1], c.latent_size)

    def forward(self, x, addl):
        B,H,W,_ = x.shape
        Cpr = self.mesh.n_pr_vars
        D = self.mesh.n_levels
        hm = self.config.hmesh
        x_sfc = x[:,:,:,self.mesh.n_pr:]

        x_sfc = torch.cat((x_sfc,addl,self.const_data),dim=-1)
        x_sfc = x_sfc.view(B, H*W, self.total_sfc)

        x_sfc = (x_sfc[:, hm.g2m_idx] * hm.g2m_wts[None,:,:,None]).sum(axis=2)
        M = x_sfc.shape[1]
        patch = self.config.patch_size
        x_sfc = x_sfc.view(B, M//(patch[1]**2), patch[1]**2 * self.total_sfc)

        x_sfc = self.conv_sfc(x_sfc)

        x_pr = x[:,:,:,:self.mesh.n_pr].view(B,H*W,Cpr,D) # really joan i swear to god
        x_pr = (x_pr[:, hm.g2m_idx, :, :] * hm.g2m_wts[None,:,:,None, None]).sum(axis=2)
        nvert = D // patch[0]
        x_pr = x_pr.view(B, M//(patch[1]**2), patch[1]**2, Cpr, nvert, patch[0]) # I hope this isn't flipped
        x_pr = x_pr.permute(0, 1, 4, 2, 3, 5)
        x_pr = x_pr.reshape(B, M//(patch[1]**2), nvert, patch[0] * patch[1]**2 * Cpr)

        x_pr = self.conv_pr(x_pr)

        x = torch.concat((x_pr, x_sfc[:, :, None, :]), dim=2)

        x += self.config.hlmesh.posemb[None]

        return x


from model_latlon.decoder import load_matrices, gather_geospatial_weights, gather_variable_weights, default_compute_loss, default_compute_errors, default_log_information, Decoder

class GNNyHealDecoder(Decoder):
    def __init__(self, mesh, config, decoder_loss_weight=None, compile=False):
        super(GNNyHealDecoder, self).__init__(self.__class__.__name__)

        self.config = config
        self.mesh = mesh
        c = config
        
        # Load norm matrices 
        _, state_norm_matrix, _ = load_matrices(mesh)
        self.register_buffer(f'state_norm_matrix', state_norm_matrix)
       
        # Gather loss weights
        geospatial_loss_weight = gather_geospatial_weights(self.mesh)
        variable_loss_weight = gather_variable_weights(self.mesh)
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight
        self.register_buffer(f'geospatial_loss_weight', geospatial_loss_weight)
        self.register_buffer(f'variable_loss_weight', variable_loss_weight)

        self.factor = 4
        # hardcoded. this is going from mesh 5 to mesh 7, 1.8 deg to 0.45 deg

        self.n_chunks = self.config.n_chunks_dec
        print("using chunks", self.n_chunks)

        with open("/fast/consts/meshes/healpix_m2g_7_c%d.pickle" % self.n_chunks, "rb") as f:
            (gathers, subgathers, edge_features) = [[torch.from_numpy(y) for y in x] for x in pickle.load(f)]

        #assert self.n_chunks == 10
        self.n_neigh = subgathers[0].shape[2]
        assert self.n_neigh == 4
        self.n_edge_features = edge_features[0].shape[-1]
        assert self.n_edge_features == 22

        def mkbufs(name, arr):
            for i in range(self.n_chunks):
                self.register_buffer(name+str(i), arr[i])

        mkbufs('gathers', gathers)
        mkbufs('subgathers', subgathers)
        mkbufs('edge_features', edge_features)
        #self.register_buffer('gathers', gathers)
        #self.register_buffer('subgathers', subgathers)
        #self.register_buffer('edge_features', edge_features)

        self.tr = IcoSlide3D(depth=self.config.encdec_tr_depth, num_heads=c.num_heads, dim=c.latent_size, hlmesh=self.config.hlmesh, checkpoint_type=self.config.checkpoint_type)

        c = self.config
        patch = c.patch_size

        self.total_sfc = self.mesh.n_sfc_vars

        self.sublatent = self.config.dec_sublatent
        self.deconv_mlp_dim = self.config.deconv_mlp_dim

        self.deconv_sfc = nn.Linear(c.latent_size, self.factor * self.factor * self.sublatent)
        self.deconv_pr = nn.Linear(c.latent_size, self.factor * self.factor * self.sublatent)

        deeper = self.config.deeper_m2g
        print("deeper?", deeper)
        gelu = nn.GELU(approximate='tanh')

        if deeper:
            self.final_sfc = nn.Sequential(
                nn.Linear((self.sublatent + self.n_edge_features)*self.n_neigh, self.deconv_mlp_dim),
                gelu,
                nn.Linear(self.deconv_mlp_dim, self.deconv_mlp_dim),
                gelu,
                nn.Linear(self.deconv_mlp_dim, self.total_sfc)
            )

            self.final_pr = nn.Sequential(
                nn.Linear((self.sublatent + self.n_edge_features)*self.n_neigh, self.deconv_mlp_dim),
                gelu,
                nn.Linear(self.deconv_mlp_dim, self.deconv_mlp_dim),
                gelu,
                nn.Linear(self.deconv_mlp_dim, self.mesh.n_pr_vars * patch[0])
            )
        else:
            self.final_sfc = nn.Sequential(
                nn.Linear((self.sublatent + self.n_edge_features)*self.n_neigh, self.deconv_mlp_dim),
                nn.GELU(),
                nn.Linear(self.deconv_mlp_dim, self.total_sfc)
            )

            self.final_pr = nn.Sequential(
                nn.Linear((self.sublatent + self.n_edge_features)*self.n_neigh, self.deconv_mlp_dim),
                nn.GELU(),
                nn.Linear(self.deconv_mlp_dim, self.mesh.n_pr_vars * patch[0])
            )


    def forward(self, x):
        x = self.tr(x)
        B, Nvert, D, C = x.shape

        nvertical = D-1
        Dout = self.mesh.n_levels

        Cpr = self.mesh.n_pr_vars
        patch = self.config.patch_size

        #x = self.tr(x)

        xp = x[:, :, :-1, :]
        #xp = self.deconv_pr(xp)
        xp = call_checkpointed(self.deconv_pr, xp.clone(), checkpoint_type=self.config.checkpoint_type)

        def do_pr_chunk(i):
            gath = getattr(self, 'gathers'+str(i))
            subgath = getattr(self, 'subgathers'+str(i))
            edge = getattr(self, 'edge_features'+str(i))

            xp2 = xp.view(B, Nvert, nvertical, self.factor * self.factor, self.sublatent)
            xp2 = xp2.permute(0, 1, 3, 2, 4)
            xp2 = xp2.reshape(B, Nvert*self.factor*self.factor, nvertical, self.sublatent)
            xp2 = xp2[:, gath][:, subgath].permute(0, 1, 2, 4, 3, 5)
            xp2 = torch.cat((xp2, edge[None, :, :, None, :, :].repeat_interleave(nvertical, dim=3).repeat_interleave(B, dim=0)), axis=5)
            xp2 = xp2.flatten(start_dim=4)
            nlat = xp2.shape[1]
            nlon = xp2.shape[2]
            assert nlon == 1440
            xp2 = self.final_pr(xp2).view(B, nlat, nlon, nvertical, Cpr, patch[0])
            xp2 = xp2.permute(0, 1, 2, 4, 3, 5).flatten(start_dim=3)

            return xp2

        x_pr = []
        for ci in range(self.n_chunks):
            x_pr.append(call_checkpointed(do_pr_chunk, ci, checkpoint_type=self.config.checkpoint_type))
        x_pr = torch.cat(x_pr, axis=1)

        xs = x[:, :, -1, :]
        #xs = self.deconv_sfc(xs)
        xs = call_checkpointed(self.deconv_sfc, xs.clone(), checkpoint_type=self.config.checkpoint_type)

        def do_sfc_chunk(i):
            gath = getattr(self, 'gathers'+str(i))
            subgath = getattr(self, 'subgathers'+str(i))
            edge = getattr(self, 'edge_features'+str(i))

            xs2 = xs.view(B, Nvert * self.factor * self.factor, self.sublatent)
            xm = xs2[:, gath][:, subgath, :]
            xm = torch.cat((xm, edge[None, :, :, :, :].repeat_interleave(B, dim=0)), axis=4)
            xm = xm.flatten(start_dim=3)
            return self.final_sfc(xm)

        x_sfc = []
        for ci in range(self.n_chunks):
            x_sfc.append(call_checkpointed(do_sfc_chunk, ci, checkpoint_type=self.config.checkpoint_type))
        x_sfc = torch.cat(x_sfc, axis=1)


        x = torch.cat((x_pr, x_sfc), axis=-1) # B,H,W,Cpr*D+Csfc
        return x

    def compute_loss(self, y_gpu, yt_gpu):
        if y_gpu.shape[0] == 1:
            return default_compute_loss(self, y_gpu, yt_gpu)
        else:
            return crps_compute_loss(self, y_gpu, yt_gpu)
    
    def compute_errors(self, y_gpu, yt_gpu, **kwargs):
        std_dic = {}
        if y_gpu.shape[0] != 1:
            std, y_gpu = torch.std_mean(y_gpu, dim=0, keepdim=True)
            w = self.geospatial_loss_weight[0,:,:,0]
            w = w / torch.sum(w)
            nm = self.state_norm_matrix
            for var_name in [p for p in self.mesh.full_varlist if "_500" in p] + self.mesh.sfc_vars:
                idx = self.mesh.full_varlist.index(var_name)
                std_dic["std_"+var_name] = (std[0, :, :, idx] * w).sum()*nm[idx]

        out = default_compute_errors(self, y_gpu, yt_gpu, **kwargs)
        out.update(std_dic)
        return out
        
    def log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
        if y_gpu.shape[0] != 1:
            for var_name in [p for p in self.mesh.full_varlist if "_500" in p] + self.mesh.sfc_vars:
                writer.add_scalar("Std"+prefix + f"_{dt}/" + var_name, rms_dict["std_"+var_name], n_step)

        default_log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir)

def afcrps_claude(y_ensemble, y_true, weights, epsilon=0.05):
    """
    Compute adjusted fair Continuous Ranked Probability Score (afCRPS)
    
    Args:
        y_ensemble: Tensor of shape [M, H, W, C] - ensemble predictions
        y_true: Tensor of shape [1, H, W, C] - ground truth
        weights: Tensor of shape [1, H, W, C] - weights for each point
        epsilon: float - adjustment parameter (default: 0.05)
    
    Returns:
        float: Mean afCRPS score
    """
    M = y_ensemble.shape[0]
    
    # Expand y_true to match ensemble shape for broadcasting
    y_true = y_true.expand(M, -1, -1, -1)
    
    # Initialize sum accumulator
    total = 0.0
    
    # Compute pairwise differences
    for j in range(M):
        # |x_j - y| term
        term1 = torch.abs(y_ensemble[j] - y_true[0])
        for k in range(M):
            if j != k:
                
                # |x_k - y| term
                term2 = torch.abs(y_ensemble[k] - y_true[0])
                
                # (1-ε)|x_j - x_k| term
                term3 = (1 - epsilon) * torch.abs(y_ensemble[j] - y_ensemble[k])
                
                # Sum the terms according to formula
                contribution = term1 + term2 - term3
                
                # Add to total
                total += (contribution*weights[0]).sum()

    # Apply coefficient and weights
    result = (1 / (2 * M * (M-1))) * total
    
    # Apply weights and compute mean
    weighted_result = result# * weights
    
    # Return mean over all dimensions
    return weighted_result#.sum()

def afcrps_claude2(y_ensemble, y_true, weights, epsilon=0.05):
    """
    Compute adjusted fair Continuous Ranked Probability Score (afCRPS)
    
    Args:
        y_ensemble: Tensor of shape [M, H, W, C] - ensemble predictions
        y_true: Tensor of shape [1, H, W, C] - ground truth
        weights: Tensor of shape [1, H, W, C] - weights for each point
        epsilon: float - adjustment parameter (default: 0.05)
    
    Returns:
        float: Mean afCRPS score
    """
    M = y_ensemble.shape[0]
    
    # Expand y_true to match ensemble shape for broadcasting
    y_true = y_true.expand(M, -1, -1, -1)
    
    # Initialize sum accumulator
    total = 0.0
    
    # Compute pairwise differences
    for j in range(M):
        # |x_j - y| term
        term1 = (torch.abs(y_ensemble[j] - y_true[0])*weights[0]).sum()
        for k in range(M):
            if j != k:
                
                # |x_k - y| term
                term2 = (torch.abs(y_ensemble[k] - y_true[0])*weights[0]).sum()
                
                # (1-ε)|x_j - x_k| term
                term3 = (1 - epsilon) * (torch.abs(y_ensemble[j] - y_ensemble[k])*weights[0]).sum()
                
                # Sum the terms according to formula
                contribution = term1 + term2 - term3
                
                # Add to total
                total += contribution

    # Apply coefficient and weights
    result = (1 / (2 * M * (M-1))) * total
    
    # Apply weights and compute mean
    weighted_result = result# * weights
    
    # Return mean over all dimensions
    return weighted_result#.sum()



def crps_compute_loss(self, y_gpu, yt_gpu):
    # Weights for loss
    combined_weight = self.geospatial_loss_weight * self.variable_loss_weight 
    weights = self.decoder_loss_weight * combined_weight / torch.sum(combined_weight)

    if 0:
        loss = torch.abs(y_gpu - yt_gpu)
        
        return torch.sum(loss * weights)
    #l = afcrps_o1_chunkH(y_gpu, yt_gpu, weights, 0.05)
    l = call_checkpointed(afcrps_claude2, y_gpu, yt_gpu, weights, 0.05, checkpoint_type=self.config.checkpoint_type)
    #print("got", l, l.item())
    return l
    import pdb; pdb.set_trace()
    loss = torch.abs(y_gpu - yt_gpu)
    
    return torch.sum(loss * weights)


class SimpleHealDecoder(Decoder):
    def __init__(self, mesh, config, decoder_loss_weight=None, compile=False):
        super(SimpleHealDecoder, self).__init__(self.__class__.__name__)

        self.config = config
        self.mesh = mesh
        c = config
        
        # Load norm matrices 
        _, state_norm_matrix, _ = load_matrices(mesh)
        self.register_buffer(f'state_norm_matrix', state_norm_matrix)
       
        # Gather loss weights
        geospatial_loss_weight = gather_geospatial_weights(self.mesh)
        variable_loss_weight = gather_variable_weights(self.mesh)
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight
        self.register_buffer(f'geospatial_loss_weight', geospatial_loss_weight)
        self.register_buffer(f'variable_loss_weight', variable_loss_weight)


        self.tr = IcoSlide3D(depth=self.config.encdec_tr_depth, num_heads=c.num_heads, dim=c.latent_size, hlmesh=self.config.hlmesh, checkpoint_type=self.config.checkpoint_type)

        self.inner = InnerSimpleHealDecoder(mesh, config)

        if compile:
            self.inner = mccompileface(self.inner)

    def forward(self, x):
        return self.inner(self.tr(x))

    def compute_loss(self, y_gpu, yt_gpu):
        return default_compute_loss(self, y_gpu, yt_gpu)
    
    def compute_errors(self, y_gpu, yt_gpu, **kwargs):
        return default_compute_errors(self, y_gpu, yt_gpu, **kwargs)
        
    def log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
        default_log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir)


class InnerSimpleHealDecoder(Decoder):
    def __init__(self, mesh, config):
        super(InnerSimpleHealDecoder, self).__init__(self.__class__.__name__)
        self.mesh = mesh
        self.config = config

        c = self.config
        patch = c.patch_size

        self.total_sfc = self.mesh.n_sfc_vars

        self.deconv_sfc = nn.Linear(c.latent_size, self.total_sfc * patch[1] * patch[1])
        self.deconv_pr = nn.Linear(c.latent_size, self.mesh.n_pr_vars * patch[0] * patch[1] * patch[1])

        self.m2g_wts = nn.Parameter(torch.ones(self.config.hmesh.m2g_idx.shape) / self.config.hmesh.m2g_idx.shape[-1])


    def forward(self, x):
        B, Nvert, D, C = x.shape

        nvertical = D-1
        Dout = self.mesh.n_levels

        Cpr = self.mesh.n_pr_vars
        patch = self.config.patch_size

        #x = self.tr(x)

        x_pr = x[:, :, :-1, :]
        x_sfc = x[:, :, -1, :]

        if self.config.patch_deltas != 0:
            x_sfc_mean = self.deconv_sfc_mean(x_sfc) # (B, Nvert, total_sfc)

        x_sfc = self.deconv_sfc(x_sfc)
        x_sfc = x_sfc.view(B, Nvert * (patch[1]**2), self.total_sfc)

        if self.config.patch_deltas != 0:
            x_sfc = x_sfc.view(B, Nvert, (patch[1]**2), self.total_sfc)
            x_sfc = self.config.patch_deltas * x_sfc + x_sfc_mean[:, :, None, :]
            x_sfc = x_sfc.view(B, Nvert * (patch[1]**2), self.total_sfc)

        x_sfc = (x_sfc[:, self.config.hmesh.m2g_idx] * self.m2g_wts[None, :, :, :, None]).sum(axis=3)

        if self.config.patch_deltas != 0:
            x_pr_mean = self.deconv_pr_mean(x_pr)
            x_pr_mean = x_pr_mean.view(B, Nvert, nvertical, Cpr, patch[0])

        x_pr = self.deconv_pr(x_pr)
        x_pr = x_pr.view(B, Nvert, nvertical, self.mesh.n_pr_vars, patch[0], patch[1], patch[1])

        if self.config.patch_deltas != 0:
            x_pr = x_pr_mean[:, :, :, :, :, None, None] + self.config.patch_deltas * x_pr

        x_pr = x_pr.permute(0, 1, 5, 6, 3, 2, 4)
        x_pr = x_pr.reshape(B, Nvert*(patch[1]**2), Cpr, Dout)
        x_pr = (x_pr[:, self.config.hmesh.m2g_idx] * self.m2g_wts[None, :, :, :, None, None]).sum(axis=3)

        x_pr = torch.flatten(x_pr, start_dim=-2) # B,H,W,Cpr*D
        x = torch.cat((x_pr, x_sfc), axis=-1) # B,H,W,Cpr*D+Csfc
        return x

    def forward_cp(self, x):
        B, Nvert, D, C = x.shape

        nvertical = D-1
        Dout = self.mesh.n_levels

        Cpr = self.mesh.n_pr_vars
        patch = self.config.patch_size

        #x = self.tr(x)


        def cp2(x_pr):
            x_pr = self.deconv_pr(x_pr)
            x_pr = x_pr.view(B, Nvert, nvertical, self.mesh.n_pr_vars, patch[0], patch[1], patch[1])

            x_pr = x_pr.permute(0, 1, 5, 6, 3, 2, 4)
            x_pr = x_pr.reshape(B, Nvert*(patch[1]**2), Cpr, Dout)
            return x_pr
        
        def cp2x(x_pr):
            #import pdb; pdb.set_trace()
            x_pr = (x_pr[:, self.config.hmesh.m2g_idx] * self.m2g_wts[None, :, :, :, None, None]).sum(axis=3)
            #x_pr = x_pr[:, self.config.hmesh.m2g_idx[:,:,0]] # nearest neighbor

            x_pr = torch.flatten(x_pr, start_dim=-2) # B,H,W,Cpr*D
            return x_pr
        x_pr = x[:, :, :-1, :]
        x_pr = call_checkpointed(cp2, x_pr.clone(), checkpoint_type=self.config.checkpoint_type)
        x_pr = call_checkpointed(cp2x, x_pr.clone(), checkpoint_type=self.config.checkpoint_type)

        def cp1(x_sfc):
            x_sfc = self.deconv_sfc(x_sfc)
            x_sfc = x_sfc.view(B, Nvert * (patch[1]**2), self.total_sfc)

            x_sfc = (x_sfc[:, self.config.hmesh.m2g_idx] * self.m2g_wts[None, :, :, :, None]).sum(axis=3)
            return x_sfc
        x_sfc = x[:, :, -1, :]
        x_sfc = call_checkpointed(cp1, x_sfc.clone(), checkpoint_type=self.config.checkpoint_type)


        x = torch.cat((x_pr, x_sfc), axis=-1) # B,H,W,Cpr*D+Csfc
        return x

if __name__ == '__main__':
    torch.manual_seed(0)
    hmesh = HealMesh(8)

    BACH = True
    BACH = False


    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    # '034_sstk', 
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', # these are both input and output
                '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    from utils import *


    import meshes
    if BACH:
        imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_joank)
        omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_joank)
    else:
        imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
        omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)


    cuda = torch.device("cuda")
    from model_latlon.top import ForecastModel, ForecastModelConfig
    from model_latlon.encoder import SimpleConvEncoder
    from model_latlon.decoder import SimpleConvDecoder
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 1024,
        pr_depth = 10,
        encdec_tr_depth = 4,
        patch_size=(5 if BACH else 4, 8, 8),
        dec_sublatent=96,
        deconv_mlp_dim=512,
    )
    conf = ForecastModelConfig(
        inputs=[imesh],
        outputs=[omesh],
        latent_size = 512,
        pr_depth = 8,
        encdec_tr_depth = 2,
        patch_size=(5 if BACH else 4, 8, 8),
        dec_sublatent=64,
        deeper_m2g=True,
        deconv_mlp_dim=256,
    )
    hlmesh = HealLatentMesh(depth=5, D=6 if BACH else 5, KL=18, KD=5, B=8, dim=conf.latent_size)
    conf.hmesh = hmesh.to(cuda)
    conf.hlmesh = hlmesh.to(cuda)
    conf.checkpoint_type = "matepoint"

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6

    if 1:
        enc = SimpleHealEncoder(imesh, conf, compile=False)
        tr = IcoSlide3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, hlmesh=hlmesh, checkpoint_type=conf.checkpoint_type)
        #dec = SimpleHealDecoder(omesh, conf, compile=True)
        dec = GNNyHealDecoder(omesh, conf, compile=False)

    else:
        enc = SimpleConvEncoder(imesh, conf)
        tr = SlideLayers3D(dim=conf.latent_size, depth=conf.pr_depth, num_heads=conf.num_heads, window_size=(5,5,5), checkpoint_type=conf.checkpoint_type)
        dec = SimpleConvDecoder(omesh, conf)
    
    print("enc", count_parameters(enc))
    print("tr", count_parameters(tr))
    print("dec", count_parameters(dec))

    tr = tr.to(cuda)
    enc = enc.to(cuda)
    dec = dec.to(cuda)

    t0 = torch.tensor([int((datetime(1997, 6, 21)-datetime(1970,1,1)).total_seconds())], dtype=torch.int64).to(cuda)


    with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
        for i in range(7):
            torch.manual_seed(i)
            if BACH: dummy = torch.randn(1, 720, 1440, 5*25+4+len(extra_output), dtype=torch.float16).to(cuda)
            else: dummy = torch.randn(1, 720, 1440, 5*16+4+len(extra_output), dtype=torch.float16).to(cuda)
            #if i == 0: torch.cuda.memory._record_memory_history(max_entries=100000)
            torch.cuda.synchronize()
            _t0 = time.time()
            dummy += i
            t0 += i
            lat = 0
            e = enc(dummy, t0)
            lat += torch.mean(e**2)
            d0 = dec(e)
            #e = dummyL
            #memprint("post enc")
            p6 = tr(e)
            lat += torch.mean(p6**2)
            d6 = dec(p6)
            #memprint("p6")
            p24 = tr(tr(tr(p6)))
            lat += torch.mean(p24**2)
            d24 = dec(p24)
            if i >= 3:
                p36 = tr(tr(p24))
                lat += torch.mean(p36**2)
                d36 = dec(p36)
                p48 = tr(tr(p36))
                lat += torch.mean(p48**2)
                d48 = dec(p48)
            else:
                d36 = d48 = torch.tensor(0.0).to(cuda)
            #memprint("total")
            #loss = dec.compute_loss(d0, dummy) + dec.compute_loss(d6, dummy) + dec.compute_loss(d24, dummy) + 1e-4 * lat + dec.compute_loss(d36, dummy) + dec.compute_loss(d48, dummy)  
            loss = torch.mean(d0)+torch.mean(d6)+torch.mean(d24) + 1e-4 * lat + torch.mean(d36) + torch.mean(d48)
            print(loss)
            loss.backward()
            memprint("backward")
            torch.cuda.synchronize()
            print("stuff took", time.time()-_t0)
            import gc
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

            while len(matepoint.Gmatepoint_ctx) > 0:
                args, devices, checksum = matepoint.Gmatepoint_ctx.pop()
                for arg in args:
                    try: arg.to(cuda)
                    except: pass
    #torch.cuda.memory._dump_snapshot("/fast/memdump_heal.pickle")
