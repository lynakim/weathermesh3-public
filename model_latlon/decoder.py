from utils import *
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_latlon.primatives3d import EarthResBlock3d
from model_latlon.primatives2d import EarthResBlock2d
from model_latlon.data import get_constant_vars, get_additional_vars, N_ADDL_VARS
from model_latlon.codec2d import EarthConvEncoder2d, EarthConvDecoder2d
from model_latlon.codec3d import EarthConvEncoder3d, EarthConvDecoder3d
from model_latlon.transformer3d import SlideLayers3D, posemb_sincos_3d, add_posemb, tr_pad, tr_unpad
from model_latlon.transformer2d import Natten2DTransformerBlock, SlideLayers2D
from model_latlon.primatives2d import southpole_pad2d, call_checkpointed, print_total_params, southpole_unpad2d
from model_latlon.primatives3d import southpole_pad3d, southpole_unpad3d, earth_pad3d
from eval import eval_rms_train, eval_rms_point
from train_fn import save_img_with_metadata
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage

# General decoder type (on the edge of being boilerplate, but definitely necessary as it prevents functions from not being defined)
class Decoder(nn.Module):
    def __init__(self, decoder_name):
        super(Decoder, self).__init__()
        self.decoder_name = decoder_name
        
        self.default_decoder_loss_weight = 1
        self._check_log_information()
    
    # When implementing a loss function, use this loss weighing heirarchy:
    # See examples in decoders defined below for more information
    # 
    #  1. Delta Time Weight [Happens in Training Loop]
    #     Exists as a dictionary in the training class
    #     NOT RELEVANT HERE
    #
    #       2. Decoder Loss Weight [Happens in Decoder] 
    #          Exists as a scalar member variable for each decoder (see default above)
    #          RELEVANT HERE
    # 
    #           3. Variable Weights [Happens in Decoder]
    #              Loaded from the json: /fast/consts/variable_weights.json
    #                   Loading must happen at Decoder initialization to avoid redundant loading (see examples below)
    #                   Loaded tensor must be converted to a registered buffer to be used effectively in the model / GPU (see examples below)
    #              RELEVANT HERE 
    # 
    #               4. Geospatial Weights [Happens in Decoder]
    #                  Computed based on the Mesh object
    #                       Computed once at Decoder initialization to avoid redundant computation (see examples below)
    #                       Computed tensor must be converted to a registered buffer to be used effectively in the model / GPU (see examples below)
    #                  RELEVANT HERE (but optional depending on the type of Mesh used)
    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError(f"You are required to implement a compute_loss function for your decoder {self.decoder_name}")
    
    def compute_errors(self, *args, **kwargs):
        raise NotImplementedError(f"You are required to implement a compute_errors function for your decoder {self.decoder_name}")
    
    # Function used to log information about the decoder to tensorboard
    def log_information(self, *args, **kwargs):
        return 
    
    # Checks if the subclass (a specific decoder) has overwritten the log_information function or not
    def _check_log_information(self):
        subclass_method = getattr(self.__class__, 'log_information', None)
        parent_method = getattr(Decoder, 'log_information', None)
        if subclass_method is parent_method: print(ORANGE(f"WARNING: log_information function is not defined for {self.decoder_name}"))
    
    # Also conceivably a separate train / eval forward() function could be implemented here, 
    # (mostly for regional TC decoding to make decode() function in top.py more readable / general)
    

def pole_hack(x):
    # this function exists because no matter what we tried, poles consistently have annoying artifacts.
    # this is a hacky solution that just copies in values from near the pole to the poles.
    x[:,0,:,:] = x[:,3,:,:]
    x[:,-1,:,:] = x[:,-4,:,:]
    return x

# This is on life support
# All we need from this is load_state_norm
# And we need to rewrite load_state_norm to take norm from norm.json
def load_matrices(mesh):
    state_norm, state_norm_matrix = load_state_norm(mesh.wh_lev, mesh)
    
    # nan_mask is only False for nan values eg. all land for sstk (like the last few rows)
    nan_mask_dict = pickle.load(open(f"{CONSTS_PATH}/nan_mask.pkl", 'rb'))
    nan_mask = np.ones((len(mesh.lats), len(mesh.lons), mesh.n_vars), dtype=bool)
    # Apply the appropriate mask for each variable
    for i, var_name in enumerate(mesh.sfc_vars):
        if var_name in nan_mask_dict:
            nan_mask[:,:,mesh.n_pr + i] = nan_mask_dict[var_name]
    
    return state_norm, torch.from_numpy(state_norm_matrix), nan_mask

# Gathers variable weights from appropriate json
def gather_variable_weights(mesh):
    default_variable_weight = 2.5
    
    variable_weights = json.load(open(f"norm/variable_weights.json", 'r'))
    
    # Output should be a tensor with shape (n_pr_vars * n_levels + n_sfc_vars)
    output_weights = []
    
    for var_name in mesh.pressure_vars:
        if var_name not in variable_weights: raise Exception(f"Pressure variable {var_name} not found in variable_weights.json. You are required to place something there.")
        assert isinstance(variable_weights[var_name], list), f"Pressure variable {var_name}'s variable weights must be a list in variable_weights.json"
        for level in mesh.levels:
            variable_level_index = variable_weights['levels'].index(level)
            variable_weight = variable_weights[var_name][variable_level_index]
            output_weights.append(variable_weight)
    
    for var_name in mesh.sfc_vars:
        if 'bucket' in var_name: continue
        if var_name not in variable_weights: 
            variable_weight = default_variable_weight # Default value
            assert var_name != 'zeropad', "im trying to load a weight for zeropad, this is sus and it feels like ur using the wrong mesh. https://chat.windbornesystems.com/#narrow/stream/201-tech-dlnwp/topic/loss.20weights/near/6939952"
            print(ORANGE(f"😵‍💫😵‍💫😵‍💫 ACHTUNG!!! No variable weight found for {var_name}. Defaulting to {default_variable_weight}. Make sure you want to do this batman..."))
            #raise Exception(f"Surface variable {var_name} not found in variable_weights.json. You are required to place something there.")
        else:
            variable_weight = variable_weights[var_name]
        assert not isinstance(variable_weight, list), f"Surface variable {var_name} must be a scalar value in variable_weights.json"
        output_weights.append(variable_weight)
        
    # Put it in a (B, N1, N2, D) shape to make broadcasting more obvious
    output_weights = torch.tensor(output_weights)
    return output_weights[np.newaxis, np.newaxis, np.newaxis, :]

def gather_geospatial_weights(mesh):
    _, Lats = np.meshgrid(mesh.lons, mesh.lats)
     
    def calc_geospatial_weight(lats):
        F = torch.FloatTensor
        weights = np.cos(lats * np.pi/180)

        # Where we want the parabola to start for the pole weights
        boundary = 50 
        # Weight at poles
        top_of_parabola = 0.3

        progress = np.arange(boundary) / (boundary - 1) # Weights for top and bottom pixels
        parabola = top_of_parabola * (1 - progress) ** 2
        weights[:boundary] += parabola[:, np.newaxis]
        weights[-boundary:] += parabola[::-1, np.newaxis]

        return F(weights)

    output_weights = calc_geospatial_weight(Lats)
    
    # Put it in a (B, N1, N2, D) shape to make broadcasting more obvious
    return output_weights[np.newaxis, :, :, np.newaxis]

# Loss function typically used for decoders operating on LatLonGrids
# Written here to reduce code duplication 
def default_compute_loss(self, y_gpu, yt_gpu):
    # Actual loss
    loss = torch.abs(y_gpu[..., :self.mesh.num_rms_vars] - yt_gpu[..., :self.mesh.num_rms_vars])
    
    # Weights for loss
    combined_weight = self.geospatial_loss_weight * self.variable_loss_weight 
    weights = self.decoder_loss_weight * combined_weight / torch.sum(combined_weight)
    
    loss = torch.sum(loss * weights)

    if len(self.mesh.bucket_vars) > 0:
        sW = torch.sum(self.geospatial_loss_weight)
        for idx, name in self.mesh.bucket_vars:
            NB = len(self.mesh.precip_buckets)
            B, H, W = y_gpu.shape[:3]
            preds = y_gpu[..., idx:idx+NB].view(B*H*W, NB)
            assert preds.shape[-1] == NB
            actuals = yt_gpu[..., idx:idx+NB].view(B*H*W, NB)
            assert actuals.shape[-1] == NB
            #print("logit rms", torch.sqrt(torch.mean(torch.square(preds.float()))))
            xe = F.cross_entropy(preds, actuals, reduction='none').view(B, H, W)
            loss += self.config.cross_entropy_weight * torch.sum(xe * self.geospatial_loss_weight[:,:,:,0]) / sW
    return loss


# This is also on life support
# Can further simplify this by removing DDP and random_time_subset
# Check out https://github.com/windborne/deep/pull/28 for more details
# Compute errors typically used for decoders operating on LatLonGrids
# Written here to reduce code duplication
def default_compute_errors(self, y_gpu, yt_gpu, trainer=None):
    B, N1, N2, D = y_gpu.shape
    nL = self.mesh.n_levels
    nP = self.mesh.n_pr_vars
    nPL = self.mesh.n_pr; assert nPL == nPL
    nS = self.mesh.n_sfc_vars - self.mesh.n_bucket_vars
    eval_shape = (B,N1,N2,nP,nL)
    to_eval = lambda z : z[...,:nPL].view(eval_shape)
    pred = to_eval(y_gpu)
    actual = to_eval(yt_gpu)
    weight = self.geospatial_loss_weight.squeeze()
    
    pnorm = self.state_norm_matrix[:nPL].view(nP, nL)
    ddp_reduce = trainer.DDP and not trainer.data.config.random_timestep_subset
    #print("pred actual", pred.device, actual.device)
    rms = eval_rms_train(pred, actual, pnorm, weight, keys=self.mesh.pressure_vars, by_level=True, stdout=False, ddp_reduce=ddp_reduce, mesh=self.mesh)

    eval_shape = (B,N1,N2,nS)
    to_eval = lambda z : z[...,nPL:self.mesh.num_rms_vars].view(eval_shape)
    pred = to_eval(y_gpu)
    actual = to_eval(yt_gpu)
    pred = pred.to(torch.float32)
    actual = actual.to(torch.float32)
    pnorm = self.state_norm_matrix[nPL:]
    rms.update(eval_rms_train(pred, actual, pnorm, weight, keys=self.mesh.sfc_vars[:nS], stdout=False, ddp_reduce=ddp_reduce, mesh=self.mesh))

    for idx, name in self.mesh.bucket_vars:
        pred = y_gpu[..., idx:idx+len(self.mesh.precip_buckets)]
        pred = torch.argmax(pred, dim=-1)
        actual = yt_gpu[..., idx:idx+len(self.mesh.precip_buckets)]
        actual = torch.argmax(actual, dim=-1)
        correct = (pred == actual).sum()/torch.numel(pred)
        rms["accuracy_" + name] = correct * 100
 
    return rms

# Log information typically used for decoders operating on LatLonGrids
# Written here to reduce code duplication
def default_log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
    # Log rms 
    for var_name in self.mesh.pressure_vars + self.mesh.sfc_vars:
        if 'bucket' in var_name: continue
        writer.add_scalar(prefix + f"_{dt}/" + var_name, rms_dict[var_name], n_step)
    for var_name in self.mesh.pressure_vars:
        name500 = var_name + "_500"
        writer.add_scalar(prefix + f"_{dt}/" + name500, rms_dict[name500], n_step)

    for idx, name in self.mesh.bucket_vars:
        writer.add_scalar(prefix + f"_{dt}/Accuracy_" + name, rms_dict["accuracy_" + name], n_step)
    
    # Save image
    if img_dir is not None:
        os.makedirs(img_dir, exist_ok=True)
        for index, var in enumerate(self.mesh.sfc_vars):
            path = os.path.join(img_dir, f'step={n_step},var={var},dt={dt},_neovis.png')
            a, p = y_gpu[0,:,:,index],yt_gpu[0,:,:,index]
            img = torch.vstack((a,p)).cpu().detach().numpy()
            save_img_with_metadata(path, img)
            diff = (a - p).cpu().detach().numpy()
            diff_path = os.path.join(img_dir, f'step={n_step},var={var}_diff,dt={dt},_neovis.png')
            save_img_with_metadata(diff_path, diff)

class SimpleConvPlusDecoder(Decoder):
    def __init__(self, mesh, config, decoder_loss_weight=None):
        super(SimpleConvPlusDecoder, self).__init__(self.__class__.__name__)
        self.mesh = mesh
        self.config = config
        
        # Load norm matrices 
        _, state_norm_matrix, _ = load_matrices(mesh)
        self.register_buffer(f'state_norm_matrix', state_norm_matrix)
       
        # Gather loss weights
        geospatial_loss_weight = gather_geospatial_weights(self.mesh)
        variable_loss_weight = gather_variable_weights(self.mesh)
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight
        self.register_buffer(f'geospatial_loss_weight', geospatial_loss_weight)
        self.register_buffer(f'variable_loss_weight', variable_loss_weight)
        
        c = config
        assert c.tr_embedding != 'sincos', "sincos is not supported for simple conv plus decoder"

        self.px_latent_pr = self.mesh.n_pr_vars*2
        self.px_latent_sfc = self.mesh.n_sfc_vars*3

        self.tr = SlideLayers3D(dim=c.latent_size, depth=c.encdec_tr_depth, num_heads=c.num_heads, window_size=c.window_size, 
                                embedding_module=c.embedding_module, checkpoint_type=c.checkpoint_type, harebrained=c.harebrained)

        self.deconv_pr = nn.ConvTranspose3d(out_channels=self.px_latent_pr, in_channels=c.latent_size, kernel_size=c.patch_size, stride=c.patch_size)
        self.deconv_sfc = nn.ConvTranspose2d(out_channels=self.px_latent_sfc, in_channels=c.latent_size, kernel_size=c.patch_size[1:], stride=c.patch_size[1:])

        self.pr_resblock = EarthResBlock3d(self.px_latent_pr, self.mesh.n_pr_vars, intermediate_channels=self.px_latent_pr)
        self.sfc_resblock = EarthResBlock2d(self.px_latent_sfc, self.mesh.n_sfc_vars, intermediate_channels=self.px_latent_sfc)

    def forward(self, x):
        B,D,H,Wp,C = x.shape

        x = self.tr(x)
        x = tr_unpad(x, self.config.window_size)
        
        x = x.permute(0,4,1,2,3) # -> B,C,D,H,W for convs

        x_pr = x[:, :, :-1, :, :]
        x_sfc = x[:, :, -1, :, :]

        def ch1(x_pr,x_sfc):
            x_pr = self.deconv_pr(x_pr)
            x_sfc = self.deconv_sfc(x_sfc)
            x_pr = self.pr_resblock(x_pr)
            x_sfc = self.sfc_resblock(x_sfc)
            return x_pr,x_sfc

        x_pr,x_sfc = call_checkpointed(ch1,x_pr,x_sfc,checkpoint_type=self.config.checkpoint_type)

        x_sfc = x_sfc.permute(0, 2, 3, 1) # B,Csfc,H,W -> B,H,W,Csfc
        x_pr = x_pr.permute(0, 3, 4, 1, 2) # B,Cpr,D,H,W -> B,H,W,Cpr,D
        x_pr = torch.flatten(x_pr, start_dim=-2) # B,H,W,Cpr*D
        x = torch.cat((x_pr, x_sfc), axis=-1) # B,H,W,Cpr*D+Csfc
        return x
    
    def compute_loss(self, y_gpu, yt_gpu):
        return default_compute_loss(self, y_gpu, yt_gpu)
    
    def compute_errors(self, y_gpu, yt_gpu, **kwargs):
        return default_compute_errors(self, y_gpu, yt_gpu, **kwargs)
        
    def log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
        default_log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir)

class StackedConvPlusDecoder(Decoder):
    def __init__(self, mesh, config, decoder_loss_weight=None):
        super(StackedConvPlusDecoder, self).__init__(self.__class__.__name__)
        self.mesh = mesh
        self.config = config
        
        # Load norm matrices 
        _, state_norm_matrix, _ = load_matrices(mesh)
        self.register_buffer(f'state_norm_matrix', state_norm_matrix)
       
        # Gather loss weights
        geospatial_loss_weight = gather_geospatial_weights(self.mesh)
        variable_loss_weight = gather_variable_weights(self.mesh)
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight
        self.register_buffer(f'geospatial_loss_weight', geospatial_loss_weight)
        self.register_buffer(f'variable_loss_weight', variable_loss_weight)
        
        c = config

       # assert c.tr_embedding != 'sincos', "sincos is not supported for stacked conv plus decoder"
        self.tr = SlideLayers3D(dim=c.latent_size, depth=c.encdec_tr_depth, num_heads=c.num_heads, window_size=c.window_size, 
                                embedding_module=c.embedding_module, checkpoint_type=c.checkpoint_type, harebrained=c.harebrained)

        assert c.latent_size % 32 == 0, f"latent size {c.latent_size} must be divisible by 32"
        assert c.patch_size == (4,8,8), f"patch size {c.patch_size} must be (4,8,8)"

        # First stage - patch size (2,4,4)
        l_pr_1 = c.latent_size//32  # 32 = 2*4*4
        l_sfc_1 = c.latent_size//16  # 16 = 4*4
        self.deconv_pr_1 = nn.ConvTranspose3d(out_channels=l_pr_1, in_channels=c.latent_size, kernel_size=(2,4,4), stride=(2,4,4))
        self.deconv_sfc_1 = nn.ConvTranspose2d(out_channels=l_sfc_1, in_channels=c.latent_size, kernel_size=(4,4), stride=(4,4))
        self.resblock_pr_1 = EarthResBlock3d(l_pr_1, l_pr_1, intermediate_channels=l_pr_1, use_pole_convs=self.config.use_pole_convs)
        self.resblock_sfc_1 = EarthResBlock2d(l_sfc_1, l_sfc_1, intermediate_channels=l_sfc_1, use_pole_convs=self.config.use_pole_convs)

        self.resblock_pr_1_2 = EarthResBlock3d(l_pr_1, l_pr_1, intermediate_channels=l_pr_1, use_pole_convs=self.config.use_pole_convs)
        self.resblock_sfc_1_2 = EarthResBlock2d(l_sfc_1, l_sfc_1, intermediate_channels=l_sfc_1, use_pole_convs=self.config.use_pole_convs)

        # Second stage - patch size (2,2,2)
        l_pr_2 = self.mesh.n_pr_vars * 2
        l_sfc_2 = self.mesh.n_sfc_vars * 3
        self.deconv_pr_2 = nn.ConvTranspose3d(out_channels=l_pr_2, in_channels=l_pr_1, kernel_size=(2,2,2), stride=(2,2,2))
        self.deconv_sfc_2 = nn.ConvTranspose2d(out_channels=l_sfc_2, in_channels=l_sfc_1, kernel_size=(2,2), stride=(2,2))
        self.resblock_pr_2 = EarthResBlock3d(l_pr_2, self.mesh.n_pr_vars, intermediate_channels=l_pr_2, use_pole_convs=self.config.use_pole_convs)
        self.resblock_sfc_2 = EarthResBlock2d(l_sfc_2, self.mesh.n_sfc_vars, intermediate_channels=l_sfc_2, use_pole_convs=self.config.use_pole_convs)

    def forward(self, x):
        B,D,H,Wp,C = x.shape

        x = self.tr(x)
        x = tr_unpad(x, self.config.window_size)
        
        x = x.permute(0,4,1,2,3) # -> B,C,D,H,W for convs
        x_pr = x[:, :, :-1, :, :]
        x_sfc = x[:, :, -1, :, :]

        # First stage
        def ch1(x_pr, x_sfc):
            x_pr = self.deconv_pr_1(x_pr)
            x_sfc = self.deconv_sfc_1(x_sfc)
            x_pr = self.resblock_pr_1(x_pr)
            x_sfc = self.resblock_sfc_1(x_sfc)
            x_pr = self.resblock_pr_1_2(x_pr)
            x_sfc = self.resblock_sfc_1_2(x_sfc)
            return x_pr, x_sfc
        x_pr, x_sfc = call_checkpointed(ch1, x_pr, x_sfc,checkpoint_type=self.config.checkpoint_type)

        # Second stage
        def ch2(x_pr, x_sfc):
            x_pr = self.deconv_pr_2(x_pr)
            x_sfc = self.deconv_sfc_2(x_sfc)
            x_pr = self.resblock_pr_2(x_pr)
            x_sfc = self.resblock_sfc_2(x_sfc)
            return x_pr, x_sfc
        x_pr, x_sfc = call_checkpointed(ch2, x_pr, x_sfc,checkpoint_type=self.config.checkpoint_type)

        x_sfc = x_sfc.permute(0, 2, 3, 1) # B,Csfc,H,W -> B,H,W,Csfc
        x_pr = x_pr.permute(0, 3, 4, 1, 2) # B,Cpr,D,H,W -> B,H,W,Cpr,D
        x_pr = torch.flatten(x_pr, start_dim=-2) # B,H,W,Cpr*D
        x = torch.cat((x_pr, x_sfc), axis=-1) # B,H,W,Cpr*D+Csfc
        if not self.training:
            x = pole_hack(x)
        return x

    def compute_loss(self, y_gpu, yt_gpu):
        return default_compute_loss(self, y_gpu, yt_gpu)
    
    def compute_errors(self, y_gpu, yt_gpu, trainer):
        return default_compute_errors(self, y_gpu, yt_gpu, trainer)

        
    def log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
        default_log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir)

class SimpleConvDecoder(Decoder):
    def __init__(self, mesh, config, decoder_loss_weight=None):
        super(SimpleConvDecoder, self).__init__(self.__class__.__name__)
        self.mesh = mesh
        self.config = config
        
        # Load norm matrices 
        _, state_norm_matrix, _ = load_matrices(mesh)
        self.register_buffer(f'state_norm_matrix', state_norm_matrix)
       
        # Gather loss weights
        geospatial_loss_weight = gather_geospatial_weights(self.mesh)
        variable_loss_weight = gather_variable_weights(self.mesh)
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight
        self.register_buffer(f'geospatial_loss_weight', geospatial_loss_weight)
        self.register_buffer(f'variable_loss_weight', variable_loss_weight)
        
        self.deconv_pr = nn.ConvTranspose3d(out_channels=self.mesh.n_pr_vars, in_channels=self.config.latent_size, kernel_size=self.config.simple_decoder_patch_size, stride=self.config.simple_decoder_patch_size)
        self.deconv_sfc = nn.ConvTranspose2d(out_channels=self.mesh.n_sfc_vars, in_channels=self.config.latent_size, kernel_size=self.config.simple_decoder_patch_size[1:], stride=self.config.simple_decoder_patch_size[1:])
        self.tr = SlideLayers3D(dim=self.config.latent_size, depth=self.config.encdec_tr_depth, num_heads=self.config.num_heads, window_size=self.config.window_size, 
                                harebrained=self.config.harebrained, embedding_module=self.config.embedding_module, checkpoint_type=self.config.checkpoint_type)

    def forward(self, x):
        B,D,H,Wp,C = x.shape

        x = self.tr(x)
        x = tr_unpad(x, self.config.window_size)
        
        x = x.permute(0,4,1,2,3) # -> B,C,D,H,W for convs

        x_pr = x[:, :, :-1, :, :]
        x_sfc = x[:, :, -1, :, :]

        x_pr = self.deconv_pr(x_pr)
        x_sfc = self.deconv_sfc(x_sfc)

        x_sfc = x_sfc.permute(0, 2, 3, 1) # B,Csfc,H,W -> B,H,W,Csfc
        x_pr = x_pr.permute(0, 3, 4, 1, 2) # B,Cpr,D,H,W -> B,H,W,Cpr,D
        x_pr = torch.flatten(x_pr, start_dim=-2) # B,H,W,Cpr*D
        x = torch.cat((x_pr, x_sfc), axis=-1) # B,H,W,Cpr*D+Csfc
        return x

    def compute_loss(self, y_gpu, yt_gpu):
        return default_compute_loss(self, y_gpu, yt_gpu)
    
    def compute_errors(self, y_gpu, yt_gpu, **kwargs):
        return default_compute_errors(self, y_gpu, yt_gpu, **kwargs)
        
    def log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
        default_log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir)

class ResConvDecoder(Decoder):
    def __init__(self, mesh, config, fuck_the_poles=False, decoder_loss_weight=None):
        super(ResConvDecoder, self).__init__(self.__class__.__name__)
        self.mesh = mesh
        self.config = config
        
        # Load norm matrices 
        _, state_norm_matrix, _ = load_matrices(mesh)
        self.register_buffer(f'state_norm_matrix', state_norm_matrix)
       
        # Gather loss weights
        geospatial_loss_weight = gather_geospatial_weights(self.mesh)
        variable_loss_weight = gather_variable_weights(self.mesh)
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight
        self.register_buffer(f'geospatial_loss_weight', geospatial_loss_weight)
        self.register_buffer(f'variable_loss_weight', variable_loss_weight)
        
        self.n_convs = 3
        self.fuck_the_poles = fuck_the_poles
        data,cvars = get_constant_vars(mesh)
        nc = len(cvars)
        data = southpole_pad2d(data.unsqueeze(0).permute(0,3,1,2))
        self.register_buffer('const_data_0', data)
        """
        for i in range(3):
            data = southpole_pad2d(F.avg_pool2d(data, kernel_size=2, stride=2))
            self.register_buffer(f'const_data_{i+1}', data)
        """

        self.total_sfc = self.mesh.n_sfc_vars + N_ADDL_VARS + len(cvars)

        self.tr = SlideLayers3D(
            dim=self.config.latent_size, 
            depth=self.config.encdec_tr_depth, 
            num_heads=self.config.num_heads, 
            window_size=self.config.window_size, 
            embedding_module=self.config.embedding_module, 
            checkpoint_type=self.config.checkpoint_type
        )
        self.sfc_decoder = EarthConvDecoder2d(self.mesh.n_sfc_vars, conv_dims=[self.config.latent_size,512,192,96],skip_dims=[0,0,0,nc], affine=self.config.affine, use_pole_convs=config.use_pole_convs)

        if self.config.oldpr:
            self.pr_decoder = nn.ConvTranspose3d(out_channels=self.mesh.n_pr_vars, in_channels=self.config.latent_size, kernel_size=(4,8,8), stride=(4,8,8))
        else:
            self.pr_decoder = EarthConvDecoder3d(self.mesh.n_pr_vars, conv_dims=[self.config.latent_size]+self.config.pr_dims[::-1], affine=self.config.affine)
        assert len(self.sfc_decoder.up_layers) == self.n_convs
        #self.pr_decoder = nn.ConvTranspose3d(out_channels=self.mesh.n_pr_vars, in_channels=self.config.latent_size, kernel_size=(4,8,8), stride=(4,8,8))
        #self.sfc_decoder = nn.ConvTranspose2d(out_channels=self.mesh.n_sfc_vars, in_channels=self.config.latent_size, kernel_size=(8,8), stride=(8,8))


    def decoder_inner(self, x):
        pass

    def decoder_sfc(self, x):

        x_sfc = x.permute(0,3,1,2)
        if self.config.oldpr:
            x_sfc = (self.sfc_decoder(x_sfc,skips=[None,None,None,self.const_data_0[:,:,:720]]))
        else:
            x_sfc = southpole_unpad2d(self.sfc_decoder(x_sfc,skips=[None,None,None,self.const_data_0]))

        x_sfc = x_sfc.permute(0,2,3,1)
        return x_sfc

    def decoder_pr(self, x):
        x_pr = x.permute(0,4,1,2,3)

        if self.config.oldpr:
            x_pr = (self.pr_decoder(x_pr))
        else:
            x_pr = southpole_unpad3d(self.pr_decoder(x_pr))

        x_pr = x_pr.permute(0,3,4,1,2)
        #x_pr = torch.flatten(x_pr, start_dim=-2) # no!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return x_pr

    def forward(self, x):
        x = self.tr(x)
        _, _, _, W, _ = x.shape
        wpad = self.config.window_size[2]//2
        x = x[:,:,:,wpad:-wpad,:]

        if self.config.oldpr:
            pr = self.decoder_pr(x[:,:-1]) # TODO: maybe checkpoint this actually
        else:
            xs = []
            for i in range(self.mesh.n_levels // 4):
                #print("inp", x.shape, x[:,i:i+1].shape)
                xs.append(call_checkpointed(self.decoder_pr, x[:,i:i+1], checkpoint_type=self.config.checkpoint_type))
                #print("er", xs[-1].shape)
            pr = torch.cat(xs, axis=-1)
        pr = torch.flatten(pr, start_dim=-2)
        #xs.append(call_checkpointed(self.decoder_sfc, x[:,-1]))
        sfc = call_checkpointed(self.decoder_sfc, x[:,-1], checkpoint_type=self.config.checkpoint_type)
        #print("ersfc", xs[-1].shape)
        y = torch.cat([pr, sfc], axis=-1)
        if self.fuck_the_poles:
            y[:,0,:,:] = y[:,3,:,:]
            y[:,-1,:,:] = y[:,-4,:,:]
        #y = call_checkpointed(self.decoder_inner, x)

        return y

    def compute_loss(self, y_gpu, yt_gpu):
        return default_compute_loss(self, y_gpu, yt_gpu)
    
    def compute_errors(self, y_gpu, yt_gpu, *args, **kwargs):
        return default_compute_errors(self, y_gpu, yt_gpu, *args, **kwargs)
        
    def log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
        default_log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir)

# TC regional decoder
#   During training loop:
#       Takes in latent space and the current time step
#       Gathers TC locations from /fast using time step
#       Slices a region using location and bounding box size 
#       Trains on that region to predict max wind speed / min pressure
#   During inference loop:
#       Takes in latent space and location 
#       Slices a region using location and bound box size
#       Predicts max wind speed / min pressure  
class RegionalTCDecoder(Decoder):
    def __init__(self, mesh, config, region_radius, hidden_dim=None, kernel=(5,8,8), stride=(5,8,8), decoder_loss_weight=None):
        super(RegionalTCDecoder, self).__init__(self.__class__.__name__)
        self.mesh = mesh
        self.config = config
        self.decoder_iter = 0
        
        # Gather loss weights
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight
        
        # Declared when creating the model 
        self.region_radius = region_radius 
        self.hidden_dim = hidden_dim
        self.kernel = kernel
        self.stride = stride
        
        # Bounding box warnings / errors
        if (self.region_radius > 5): print(ORANGE("Warning: Bounding box size is excessively large, remember we're working with latent space dims here (720x1440/8)"))
        
        # Load mean / variance pickle file 
        with open('/fast/consts/normalization.pickle', 'rb') as f:
            self.maxws_mean, self.maxws_variance = (value[0] for value in pickle.load(f)['tc-maxws'])
        
        # Actual model architecture
        self.transposedConv = nn.ConvTranspose3d(
            in_channels=self.config.latent_size,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel, 
            stride=self.stride)
        # 6 is hardcoded right now, assumes 5 for pressure, 1 for sfc
        D = (8 - 1) * self.stride[0] + self.kernel[0]
        H = ((2 * self.region_radius)) * self.stride[1] + self.kernel[1]
        W = ((2 * self.region_radius)) * self.stride[2] + self.kernel[2]
        self.linear_dim = self.hidden_dim * D * H * W
        self.linear = nn.Linear(self.linear_dim, 2)
           
    # Gathers a region from the latent space using the location and bounding box
    def gather_region(self, latent_space, location):
        x_location = int(location[0] / 8)
        y_location = int(location[1] / 8)
        
        # Padding for region if we're accessing a location on the bounds of the latent space
        x_latent_space_bound = latent_space.shape[3]
        y_latent_space_bound = latent_space.shape[4]
        minx = max(x_location - self.region_radius, 0)
        maxx = min(x_location + self.region_radius + 1, x_latent_space_bound)
        miny = max(y_location - self.region_radius, 0)
        maxy = min(y_location + self.region_radius + 1, y_latent_space_bound)
        
        region = latent_space[:, 
                              :, 
                              :,
                              minx:maxx, 
                              miny:maxy,]
        
        pad_left_x = max(0, self.region_radius - x_location)
        pad_right_x = max(0, (x_location + self.region_radius + 1) - x_latent_space_bound)
        pad_left_y = max(0, self.region_radius - y_location)
        pad_right_y = max(0, (y_location + self.region_radius + 1) - y_latent_space_bound)
        
        region = F.pad(region, (pad_left_y, pad_right_y, pad_left_x, pad_right_x))
        
        assert region.shape[-2] == (2 * self.region_radius) + 1 and region.shape[-1] == (2 * self.region_radius) + 1, f"Region shape must match region's radius | {2 * self.region_radius + 1} vs {region.shape[-3]}, {region.shape[-2]} | location: {location} | pad_left_x {pad_left_x} | pad_right_x {pad_right_x} | pad_left_y {pad_left_y} | pad_right_y {pad_right_y}"
        return region
    
    # Used during evaluation to gather locations from WM decoder output
    def gather_locations(self, tc_grid):
        assert tc_grid.shape == (720, 1440), "tc_grid should be of shape (720, 1440)"
        sigma = 5
        min_storm_maxws = 40 # basic threshold, but theoretically should be 64 knots
        tc_grid = tc_grid.to(torch.float32)
        threshold = ( min_storm_maxws - self.maxws_mean ) / np.sqrt(self.maxws_variance)
        
        smoothed_grid = gaussian_filter(tc_grid.to('cpu'), sigma=sigma)
        max_filter = np.array(np.ones((3, 3)), dtype=bool)
        local_max = (smoothed_grid == ndimage.maximum_filter(smoothed_grid, footprint=max_filter)) & (smoothed_grid > threshold)
        
        x, y = np.where(local_max)
        return list(zip(x, y))
    
    # Decodes a region of the latent space to predict max wind speed / min pressure
    # Should only output 2 variables
    def decode(self, region):
        x = self.transposedConv(region)
        x = F.relu(x)
        x = torch.flatten(x)
        assert x.shape[0] == self.linear_dim, f"Shape should be {self.linear_dim}, it is {x.shape[0]}"
        return self.linear(x)
    
    def forward(self, latent_space, time=None, locations=None):
        assert time is not None or locations is not None, "Either time or locations must be provided"

        # Only add locations if locations is supplied (should only happen during evaluation)
        should_add_location = locations is not None

        if locations is None:
            date = get_date(time)
            file_name = f'/fast/proc/cyclones/locations/{date.year}{date.month:02d}/{time}.npz'
            locations = np.load(file_name)['x']
            
        if len(locations) > 0:
            latent_space = tr_unpad(latent_space, self.config.window_size)
            latent_space = latent_space.permute(0,4,1,2,3)
        
        # Run the decoder a variable number of times
        output_intensities = []
        # Note that we may be running the decoder 0 times if there are no hurricanes present
        for location in locations:
            print(GREEN(f"Fowarding TC regional decoder at {location}... | dim_latent_space: {latent_space.shape}"))
        
            region = self.gather_region(latent_space, location)
            decoded_intensity = self.decode(region)
            del region
            
            if should_add_location:
                decoded_intensity = torch.cat([decoded_intensity.to('cpu'), torch.tensor(location)])
            else:
                decoded_intensity = decoded_intensity
            
            output_intensities.append(decoded_intensity)
        
        if len(output_intensities) == 0:
            return torch.tensor([])
        
        return torch.stack(output_intensities).unsqueeze(dim=0)
    
    def compute_loss(self, y_gpu, yt_gpu):
        # Actual loss
        loss = torch.abs(y_gpu - yt_gpu)
        
        # Weights for loss
        print(ORANGE(f"Absolute loss sum: {torch.sum(loss)}"))
        decoder_loss = loss * self.decoder_loss_weight
        print(ORANGE(f"Decoder weighted loss sum: {torch.sum(decoder_loss)}"))
        
        return torch.sum(decoder_loss)
    
    def compute_errors(self, y_gpu, yt_gpu, *args, **kwargs):
        assert y_gpu.shape == yt_gpu.shape, "Prediction and target must have the same shape"
        
        rms = []
        for y, yt in zip(y_gpu.squeeze(0), yt_gpu.squeeze(0)):
            diff = y - yt
            squared_sum = torch.sum(diff**2)
            rms.append(torch.sqrt(squared_sum))
            
        self.rms, self.y_gpu, self.yt_gpu = rms, y_gpu, yt_gpu

    def log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
        assert False, "Sorry Jack, I broke this with a refactor, you need ot update it. - John"
        # Log rms 
        for error in self.rms:
            writer.add_scalar(prefix, error, self.decoder_iter)
            self.decoder_iter += 1  
            
        # Save image    
        if img_dir is not None:
            txt_path = os.path.join(img_dir, f'step={n_step},decoder=RegionalTCDecoder,dt={dt}.txt')
            with open(txt_path, 'w') as file:
                file.write(f"Prediction: \n")
                file.write(str(self.yt_gpu.to('cpu').tolist()) + "\n")
                file.write(f"Target: \n")
                file.write(str(self.y_gpu.to('cpu').tolist()))

class SimpleConvDecoder9km(Decoder):
    def __init__(self, mesh, config, decoder_loss_weight=None):
        super(SimpleConvDecoder9km, self).__init__(self.__class__.__name__)
        self.mesh = mesh
        self.config = config
        
        # Load norm matrices 
        _, state_norm_matrix, _ = load_matrices(mesh)
        self.register_buffer(f'state_norm_matrix', state_norm_matrix)
       
        # Gather loss weights
        geospatial_loss_weight = gather_geospatial_weights(self.mesh)
        variable_loss_weight = gather_variable_weights(self.mesh)
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight
        
        self.register_buffer(f'geospatial_loss_weight', geospatial_loss_weight)
        self.register_buffer(f'variable_loss_weight', variable_loss_weight)
        self.deconv_pr = nn.ConvTranspose3d(out_channels=self.mesh.n_pr_vars, in_channels=self.config.latent_size, kernel_size=self.config.patch_size_9km, stride=self.config.patch_size_9km)
        self.deconv_sfc = nn.ConvTranspose2d(out_channels=self.mesh.n_sfc_vars, in_channels=self.config.latent_size, kernel_size=self.config.patch_size_9km[1:], stride=self.config.patch_size_9km[1:])
        self.tr = SlideLayers3D(dim=self.config.latent_size, depth=self.config.encdec_tr_depth, num_heads=self.config.num_heads, window_size=self.config.window_size, harebrained=self.config.harebrained, embedding_module=self.config.embedding_module, checkpoint_type=self.config.checkpoint_type)

    def forward(self, x):
        B,D,H,Wp,C = x.shape

        x = self.tr(x)
        x = tr_unpad(x, self.config.window_size)
        
        x = x.permute(0,4,1,2,3)

        x_pr = x[:, :, :-1, :, :]
        x_sfc = x[:, :, -1, :, :]
        x_pr = self.deconv_pr(x_pr)
        x_sfc = self.deconv_sfc(x_sfc)
        assert x_pr.shape[3] == 1800, x_pr.shape
        assert x_sfc.shape[3] == 3600, x_sfc.shape

        x_sfc = x_sfc.permute(0, 2, 3, 1)
        x_pr = x_pr.permute(0, 3, 4, 1, 2)
        x_pr = torch.flatten(x_pr, start_dim=-2)
        x = torch.cat((x_pr, x_sfc), axis=-1)
        return x
    
    def compute_loss(self, y_gpu, yt_gpu):
        return default_compute_loss(self, y_gpu, yt_gpu)
    
    def compute_errors(self, y_gpu, yt_gpu, **kwargs):
        return default_compute_errors(self, y_gpu, yt_gpu, **kwargs)
        
    def log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
        default_log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir)

class RegionalSimpleConvDecoder(Decoder):
    """Like OldDecoder but applied on a region only"""
    def __init__(self, mesh, config, latent_bounds, real_bounds, decoder_loss_weight=None):
        super(RegionalSimpleConvDecoder, self).__init__(self.__class__.__name__)
        self.mesh = mesh
        self.config = config 
        
        # Load norm matrices 
        _, state_norm_matrix, _ = load_matrices(mesh)
        self.register_buffer(f'state_norm_matrix', state_norm_matrix)
       
        # Gather loss weights
        geospatial_loss_weight = gather_geospatial_weights(self.mesh)
        variable_loss_weight = gather_variable_weights(self.mesh)
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight
        self.register_buffer(f'geospatial_loss_weight', geospatial_loss_weight)
        self.register_buffer(f'variable_loss_weight', variable_loss_weight)
        
        self.latent_bounds = latent_bounds
        self.real_bounds = real_bounds
        
        # bounds are tuples (lat_min, lat_max, lon_min, lon_max), maxes non-inclusive
        assert len(latent_bounds) == 4, latent_bounds
        assert len(real_bounds) == 4, real_bounds

        self.deconv_pr = nn.ConvTranspose3d(out_channels=self.mesh.n_pr_vars, in_channels=self.config.latent_size, kernel_size=(5,8,8), stride=(5,8,8))
        self.deconv_sfc = nn.ConvTranspose2d(out_channels=self.mesh.n_sfc_vars, in_channels=self.config.latent_size, kernel_size=(8,8), stride=(8,8))
        self.tr = SlideLayers3D(dim=self.config.latent_size, depth=self.config.encdec_tr_depth, num_heads=self.config.num_heads, window_size=self.config.window_size, harebrained=self.config.harebrained)
        self.latent_bounds = latent_bounds
        self.real_bounds = real_bounds

    def forward(self, x):
        B,D,H,Wp,C = x.shape

        x = self.tr(x)
        x = tr_unpad(x, self.config.window_size)
        
        x = x.permute(0,4,1,2,3)

        lat_min, lat_max, lon_min, lon_max = self.latent_bounds
        x_pr = x[:, :, :-1, lat_min:lat_max, lon_min:lon_max]
        x_sfc = x[:, :, -1, lat_min:lat_max, lon_min:lon_max]
        x_pr = self.deconv_pr(x_pr)
        x_sfc = self.deconv_sfc(x_sfc)
        assert x_pr.shape[3] == 160, x_pr.shape
        assert x_sfc.shape[3] == 288, x_sfc.shape

        x_sfc = x_sfc.permute(0, 2, 3, 1)
        x_pr = x_pr.permute(0, 3, 4, 1, 2)
        x_pr = torch.flatten(x_pr, start_dim=-2)
        x = torch.cat((x_pr, x_sfc), axis=-1)
        return x
    
    def compute_loss(self, y_gpu, yt_gpu):
        return default_compute_loss(self, y_gpu, yt_gpu)
    
    def compute_errors(self, y_gpu, yt_gpu, **kwargs):
        return default_compute_errors(self, y_gpu, yt_gpu, **kwargs)
        
    def log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir=None):
        default_log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir)


class PointDecoder(Decoder):
    def __init__(
            self, 
            mesh, 
            config, 
            tr2d_hidden_dim=1024,
            tr2d_depth=4,
            tr2d_num_heads=32,
            tr2d_window_size=(3,3),
            inner_res=0.1,
            n_deconv_channels=256,
            n_point_vars=8,
            local_patch_size=1, # degs
            decoder_loss_weight=None,
        ):
        super(PointDecoder, self).__init__(self.__class__.__name__)
        self.mesh = mesh
        self.config = config
       
        # Gather loss weights
        # TODO: maybe there should be a variable loss weight for point vars
        self.decoder_loss_weight = decoder_loss_weight if decoder_loss_weight is not None else self.default_decoder_loss_weight

        self.inner_res = inner_res
        assert local_patch_size / inner_res == int(local_patch_size / inner_res), f"local_patch_size must be a multiple of inner_res"
        assert 2 / inner_res == int(2 / inner_res), f"2 must be a multiple of inner_res"
        k = int(2 / inner_res)
        self.deconv_kernel_size = (k, k)
        self.n_deconv_channels = n_deconv_channels
        self.tr2d_hidden_dim = tr2d_hidden_dim
        self.tr2d_depth = tr2d_depth
        self.tr2d_num_heads = tr2d_num_heads
        self.tr2d_window_size = tr2d_window_size

        self.point_vars = self.mesh.vars
        self.n_point_vars = n_point_vars
        assert n_point_vars == len(self.point_vars), f"need to update point_vars in PointDecoder class if changing n_point_vars"
        self.variable_loss_weights = [1., 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.5]
        assert len(self.variable_loss_weights) == n_point_vars, f"need to update variable_loss_weights in PointDecoder class if changing n_point_vars"
        
        self.local_patch_size = local_patch_size

        self.deconv_latent = nn.ConvTranspose2d(
            out_channels=self.n_deconv_channels // 2,
            in_channels=self.config.latent_size,
            kernel_size=(20,20),
            stride=(20,20),
        )

        k2 = k // 20
        self.deconv_latent_2 = nn.ConvTranspose2d(
            out_channels=self.n_deconv_channels,
            in_channels=self.n_deconv_channels // 2,
            kernel_size=(k2,k2),
            stride=(k2,k2),
        )

        self.deconv_25km = nn.Sequential(
            nn.ConvTranspose2d(out_channels=10, in_channels=10, kernel_size=(5,5), stride=(5,5)),
            nn.ReLU(),
            # nn.ConvTranspose2d(out_channels=10, in_channels=10, kernel_size=(5,5), stride=(5,5)),
            # nn.ReLU(),
        )

        self.tr3d = SlideLayers3D(
            dim=self.config.latent_size, 
            depth=self.config.encdec_tr_depth,
            num_heads=self.config.num_heads, 
            window_size=self.config.window_size, 
            embedding_module=self.config.embedding_module, 
            checkpoint_type=self.config.checkpoint_type
        )

        self.tr2d = SlideLayers2D(
            dim=self.tr2d_hidden_dim,
            depth=self.tr2d_depth,
            num_heads=self.tr2d_num_heads,
            window_size=self.tr2d_window_size,
            checkpoint_type=self.config.checkpoint_type
        )

        self.static_conv = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=256, kernel_size=(2,2), stride=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=self.tr2d_hidden_dim - 10 - self.n_deconv_channels, kernel_size=(3,3), stride=(3,3)),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=5, out_channels=256, kernel_size=(2,2), stride=(2,2)),
            # nn.ReLU()
        )

        self.out = nn.Linear(self.tr2d_hidden_dim, n_point_vars)

        self.load_statics()
        self.setup_norm()

        # print number of parameters of each layer
        n_params = defaultdict(int)
        for name, param in self.named_parameters():
            module_name = name.split('.')[0]
            n_params[module_name] += param.numel()
        for name, param in n_params.items():
            print(f"{name}: {param / 1e6:.2f}M")

    def setup_norm(self):
        norm = json.load(open('/fast/haoxing/deep/norm/norm.json','r'))
        norm_means, norm_stds = [], []
        for var in self.mesh.vars:
            ncar_var = self.mesh.metar_to_ncar[var]
            norm_means.append(norm[ncar_var]['mean'])
            norm_stds.append(norm[ncar_var]['std'])
        self.norm_means = torch.tensor(norm_means)
        self.norm_stds = torch.tensor(norm_stds)

    def load_statics(self):
        """ Statics for PointDecoder:
        - Elevation30: 30 arcsec elevation data
        - Modis30: 30 arcsec soil type data
        - Static_sfc: 25km static surface data: orograph, cos/sin lon/lat
        """
        
        self.Elevation30 = np.load("/huge/ignored/elevation/mn30.npy", mmap_mode='r')
        self.Modis30 = np.load("/huge/ignored/modis/2020_small2.npy", mmap_mode='r')

        lons = np.arange(0, 360, 0.25)
        lats = np.arange(90, -90, -0.25)
        lons2d, lats2d = np.meshgrid(lons, lats)

        coslons2d = np.cos(lons2d * np.pi/180)
        sinlons2d = np.sin(lons2d * np.pi/180)
        coslats2d = np.cos(lats2d * np.pi/180)
        sinlats2d = np.sin(lats2d * np.pi/180)

        lons2d_mod2 = lons2d % 2 / 2
        lats2d_mod2 = lats2d % 2 / 2

        orography = np.load("/huge/consts/topography.npy")[:720]
        orography /= np.max(orography)

        self.Static_sfc = np.stack([orography, coslons2d, sinlons2d, coslats2d, sinlats2d, lons2d_mod2, lats2d_mod2], axis=2)

    def interpget(self, src, toy, hr):
        return np.load("/huge/consts/"+'/%s/%d_%d.npy' % (src, toy, hr), mmap_mode='r')
    
    def get_rad_and_ang(self, timestamp):
        date = get_date(timestamp)
        soy = date.replace(month=1, day=1)
        toy = int((date - soy).total_seconds()/86400)
        Rad = self.interpget("neoradiation_1", toy, date.hour)
        Ang = self.interpget("solarangle_1", toy, date.hour)
        return Rad, Ang

    def get_local_patch(self, full_arr, lat, lon):
        baselat = int(np.round(lat + self.local_patch_size / 2))
        baselon = int(np.round(lon - self.local_patch_size / 2))
        baselonp = (baselon + 360) % 360
        ires = full_arr.shape[0] // 180
        idxlat = (90 - baselat) * ires
        idxlon = baselonp * ires
        assert int(self.local_patch_size * ires) == self.local_patch_size * ires, f"local_patch_size: {self.local_patch_size}, ires: {ires}"
        L = int(self.local_patch_size * ires)
        idxlon_end = (idxlon + L) % full_arr.shape[1]
        idxlat_end = idxlat + L

        if idxlat_end > full_arr.shape[0] or idxlat < 0:
            pad_top = max(0, idxlat_end - full_arr.shape[0])
            pad_bottom = max(0, -idxlat)

            sliced_arr = full_arr[max(0, idxlat):min(full_arr.shape[0], idxlat_end), :]
            sliced_arr = np.pad(
                sliced_arr,
                tuple([(pad_bottom, pad_top)] + [(0, 0)] * (len(sliced_arr.shape) - 1)),
                mode='constant', constant_values=0  
            )
        else:
            sliced_arr = full_arr[idxlat:idxlat_end]

        # Ensure longitude slicing handles wrapping correctly
        if idxlon_end > idxlon:
            local_arr = sliced_arr[:, idxlon:idxlon_end].copy()
        else:
            # Handle wrapping by concatenating the end and beginning of the array
            local_arr = np.concatenate(
                (sliced_arr[:, idxlon:], sliced_arr[:, :idxlon_end]),
                axis=1
            )
        assert local_arr.shape[:2] == (self.local_patch_size*ires, self.local_patch_size*ires), f"local_arr shape: {local_arr.shape}, baselat: {baselat}, baselon: {baselon}, idxlat: {idxlat}, idxlon: {idxlon}, ires: {ires}"
        return local_arr

    def get_statics(self, timestamp, points):
        static_types = ["Elevation30", "Modis30", "Rad", "Ang", "Static_sfc"]
        static_data = {key: [] for key in static_types}
        for lat, lon in points:
            
            Rad, Ang = self.get_rad_and_ang(timestamp)
            for static_type in static_types:
                full_arr = getattr(self, static_type) if static_type not in ["Rad", "Ang"] else locals()[static_type]
                local_arr = self.get_local_patch(full_arr, lat, lon)
                if static_type == "Modis30":
                    if lat < -65: # this used to be if baselat < -65
                        local_arr = local_arr * 0 + 15 # copied from neopointy which was copied from who knows where
                    local_arr[local_arr == -1] = 17
                    local_arr[local_arr == 0] = 17
                    local_arr -= 1
                    assert local_arr.min() >= 0 and local_arr.max() < 17
                    # # turn into one-hot
                    # local_arr = np.eye(17)[local_arr].astype(np.float16)
                elif static_type == "Elevation30":
                    local_elevs = [
                        local_arr * 0.001,
                        local_arr * 0 + np.mean(local_arr) * 0.001,
                        local_arr * 0 + np.std(local_arr) * 0.005,
                        (local_arr - np.mean(local_arr)) / (1 + np.std(local_arr)),
                    ]
                    local_arr = np.stack(local_elevs, axis=-1).astype(np.float16)
                elif static_type == "Rad":
                    local_arr = ((local_arr - 300)/400).astype(np.float16)[:,:,None]
                elif static_type == "Ang":
                    local_arr = local_arr * np.pi / 180
                    local_arr = np.stack([np.cos(local_arr), np.sin(local_arr)], axis=-1).astype(np.float16)
                elif static_type == "Static_sfc":
                    local_arr = local_arr.astype(np.float16)
                static_data[static_type].append(torch.tensor(local_arr))
        for key in static_data:
            static_data[key] = torch.stack(static_data[key], axis=0)
        return static_data

    def index_latent(self, x, latlons):
        """
        Take the slice of latent space corresponding to the points
        Latent space is (1, 8, 90, 180, hidden_dim), 2x2 degrees
        """
        B, D, H, W, C = x.shape
        assert B == 1, "assuming batch size 1"
        assert H == 90, "assuming 2 degree resolution for latent space"
        # lat index 0 starts at +90, lon index 0 starts at 0
        res = 2
        lati = np.clip(((90. - latlons[:, 0]) // res).astype(int), 0, H - 1)
        loni = (((180 // res) + (latlons[:, 1] + 180.) // res) % (360 // res)).astype(int)
        assert lati.min() >= 0 and lati.max() < H, f"lati: {lati.min()} {lati.max()}"
        assert loni.min() >= 0 and loni.max() < W, f"loni: {loni.min()} {loni.max()}"
        x = x[0, -1, lati, loni]
        return x # shape: (B, hidden_dim)
    
    def index_latent_patch(self, x, latlons):
        """
        Take a 3x3 patch of latent space centered at the given points.
        Latent space is (1, 8, 90, 180, hidden_dim), 2x2 degrees
        
        Args:
            x: Tensor of shape (1, 8, 90, 180, hidden_dim)
            latlons: Array of latitude/longitude pairs
        
        Returns:
            Tensor of shape (N, 3, 3, hidden_dim) where N is number of points
        """
        B, D, H, W, C = x.shape
        assert B == 1, "assuming batch size 1"
        assert H == 90, "assuming 2 degree resolution for latent space"
        
        # Get center indices
        res = 2
        lati = np.clip(((90. - latlons[:, 0]) // res).astype(int), 0, H - 1)
        loni = (((180 // res) + (latlons[:, 1] + 180.) // res) % (360 // res)).astype(int)
        
        # Create arrays for the 3x3 offsets
        di = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        dj = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        
        # Compute indices for each point in the 3x3 patch
        patch_lati = np.clip(lati[:, None] + di[None, :], 0, H - 1)  # Clip latitude at poles
        patch_loni = (loni[:, None] + dj[None, :]) % W  # Wrap longitude around globe
        
        # Index into the latent space
        x = x[0, -1]  # Select batch and time step
        patches = x[patch_lati, patch_loni]  # Shape: (N, 9, hidden_dim)
        
        # Reshape to (N, 3, 3, hidden_dim)
        N = latlons.shape[0]
        patches = patches.reshape(N, 3, 3, -1)
        
        return patches
    
    def patch_to_point(self, x, latlons, res, extent):
        """
        x: shape (batch, hidden_dim, n, n) where n is 3 * patch_size
        """
        patch_size = int(extent / res)
        idx_in_coarse_patch = np.floor((latlons % extent) / res).astype(int)
        idx_in_finer_patch = idx_in_coarse_patch + np.array([patch_size, patch_size])
        baselat = np.round(latlons[:, 0] + self.local_patch_size / 2).astype(int)
        baselon = np.round(latlons[:, 1] - self.local_patch_size / 2).astype(int)
        offset_lat = np.floor((baselat - latlons[:, 0]) / res).astype(int)
        offset_lon = np.floor((latlons[:, 1] - baselon) / res).astype(int)
        lati, loni = idx_in_finer_patch[:, 0], idx_in_finer_patch[:, 1]
        lati -= offset_lat
        loni -= offset_lon
        return torch.stack([x[i, :, lat:lat+patch_size, lon:lon+patch_size] for i, (lat, lon) in enumerate(zip(lati, loni))], dim=0)

    def posemb_sincos_2d(self, h, w, dim, temperature: int = 10000, dtype = torch.float32):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)

    def downsize_patch(self, x, n, n_new):
        """
        Take the center n_new x n_new patch of the n x n patch
        """
        assert n_new <= n, "downsize_patch: new size must be smaller"
        assert n % n_new == 0, "downsize_patch: new size must be a multiple of old size"
        start = (n - n_new) // 2
        return x[:,:,start:start+n_new,start:start+n_new]

    def forward(self, x, timestamp, points):
        """
        x: latent space tensor of shape (B, D, H, W, C)
        timestamp: timestamp of the current data
        points: tensor of shape (B, 2) of latlons
        """
        x = self.tr3d(x)
        x = tr_unpad(x, self.config.window_size) # shape: (1, D, H, W, C)
        x = self.index_latent_patch(x, points) # shape: (B, latent_size, 3, 3)
        x = self.deconv_latent(x.permute(0, 3, 1, 2)) # shape: (B, n_deconv_channels, 120, 120)
        x = self.patch_to_point(x, points, res=0.1, extent=2) # shape: (B, n_deconv_channels, 20, 20)
        x = self.deconv_latent_2(x)
        n = int(self.local_patch_size / self.inner_res)
        x = self.downsize_patch(x, x.shape[2], n)


        # statics shapes, B = PointDataset batch size
        # 25km: (B, 4, 4, 1+2+5) for 1 deg patches
        # 30arcsec: (B, 120, 120, 4+1) for 1 deg patches
        t0 = time.time()
        static_data = self.get_statics(timestamp, points)
        print(f"time to get statics: {time.time() - t0:.3f} s")
        for key in static_data:
            static_data[key] = static_data[key].to(x.device)
        statics_25km = torch.cat((static_data["Rad"], static_data["Ang"], static_data["Static_sfc"]), axis=-1)
        statics_25km = statics_25km.permute(0, 3, 1, 2)
        statics_25km = self.deconv_25km(statics_25km)
        statics_30arcsec = torch.cat((static_data["Elevation30"], static_data["Modis30"].unsqueeze(-1)), axis=-1)
        
        features_30arcsec = self.static_conv(statics_30arcsec.permute(0, 3, 1, 2))

        x = torch.cat([x, statics_25km, features_30arcsec], axis=1)
        x = x.permute(0, 2, 3, 1)
        posemb = self.posemb_sincos_2d(x.shape[1], x.shape[2], self.tr2d_hidden_dim, temperature=1000, dtype=x.dtype).to(x.device).view(x.shape[1], x.shape[2], self.tr2d_hidden_dim)
        x = x + posemb
        x = call_checkpointed(self.tr2d, x, checkpoint_type=self.config.checkpoint_type)
        x = x.permute(0, 3, 1, 2)
        #assert x.shape[2] == 8, "doing stuff in 2x2 degree patches at 0.25 deg resolution"
        # assert x.shape[2] == 10
        # x = x[:,:,4:6,4:6] # get the center
        n = x.shape[2]
        c = n // 2 - 1
        x = x[:,:,c,c]
        x = self.out(x).squeeze()
        print("output shape: ", x.shape)
        return x
    
    def compute_loss(self, y_gpu, yt_gpu, point_weights):
        #yt_gpu, point_weights = yt_gpu[:self.CAP], point_weights[:self.CAP]
        y_gpu = y_gpu.squeeze() # TODO I don't actually understand why this needs squeezing too
        yt_gpu = yt_gpu.squeeze() # TODO squeeze is there because collate_fn
        #variable_weights = torch.ones(self.n_point_vars, device=y_gpu.device)
        variable_weights = torch.tensor(self.variable_loss_weights, device=y_gpu.device)
        weights = point_weights.unsqueeze(-1) * variable_weights
        self.loss = F.mse_loss(y_gpu * weights, yt_gpu * weights, reduction='none').mean(axis=0)
        return self.loss.sum()
    
    def compute_errors(self, y_gpu, yt_gpu, trainer, point_weights):
        #yt_gpu, point_weights = yt_gpu[:self.CAP], point_weights[:self.CAP]
        yt_gpu = yt_gpu.squeeze() # TODO squeeze is there because collate_fn
        ddp_reduce = trainer.DDP and not trainer.data.config.random_timestep_subset
        with torch.no_grad():
            self.nan_fractions = torch.isnan(yt_gpu).float().mean(axis=0)
            self.n_points = yt_gpu.shape[0]
            self.rms = eval_rms_point(y_gpu, yt_gpu, point_weights, self.norm_means, self.norm_stds, ddp_reduce=ddp_reduce)
        print(f"temp rms: {self.rms[0]}")
        return
           
    def log_information(self, y_gpu, yt_gpu, rms_dict, writer, dt, n_step, prefix, img_dir):
        writer.add_scalar(f"zPointData_{dt}/n_points", self.n_points, n_step)
        for i, var_name in enumerate(self.point_vars):
            rms, loss, nan_frac = self.rms[i], self.loss[i], self.nan_fractions[i]
            writer.add_scalar(prefix + f"_{dt}/" + var_name, rms, n_step)
            writer.add_scalar(f"Loss_{dt}/" + var_name, loss, n_step)
            writer.add_scalar(f"zPointData_{dt}/nan_frac/" + var_name, nan_frac, n_step)
        
        assert "extra" not in self.__dict__, "extra should not be in self.__dict__ for PointDecoder"

if __name__ == "__main__":
    import model_latlon.top as top
    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), input_levels=levels_medium, levels=levels_ecm1)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, input_levels=levels_medium, levels=levels_ecm1)
    config = top.ForecastModelConfig(
        imesh,
        latent_size = 1280,
        pr_dims = [32,64,256],
        patch_size = (4,8,8),
        affine = True,
        pr_depth = 10,
        encdec_tr_depth = 4,
        oldpr = False,
        tr_embedding = 'sincos', 
    )
    decoder = top.StackedConvPlusDecoder(omesh,config)
    print(decoder)
    print_total_params(decoder)
