from utils import *
from data import *
from model_latlon_3d import ForecastStepBase, BasicLayer3D, ForecastStepDecoder, ForecastStepEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy




class ForecastStepAdapter(ForecastStepBase):
    def __init__(self,base_model):
        super().__init__()
        self.base_model = base_model
        self.const_data = base_model.const_data 
        self.config = base_model.config
        c = self.config ; mesh = c.mesh
        self.conv = nn.Conv3d(in_channels=mesh.n_pr_vars, out_channels=c.conv_dim, kernel_size=c.patch_size, stride=c.patch_size)
        self.conv_sfc = nn.Conv2d(in_channels=c.total_sfc, out_channels=c.conv_dim, kernel_size=c.patch_size[1:], stride=c.patch_size[1:])
        self.enc_swin = BasicLayer3D(dim=c.hidden_dim, input_resolution=c.resolution, depth=c.adapter_swin_depth, num_heads=c.num_heads, window_size=c.window_size, FLASH=c.FLASH, drop_path=c.drop_path, checkpoint_every=c.checkpoint_every, checkpointfn=c.checkpointfn)

        self.decoder = ForecastStepDecoder(dt=0, config=self.config)

        #self.conv = deepcopy(base_model.conv)
        #self.conv_sfc = deepcopy(base_model.conv_sfc)

    def forward(self,x):
        c = self.config
        
        assert self.config.lat_compress == False

        xpr,xsfc = self.breakup_pr_sfc(x)
        xpr_conv = self.conv(xpr)
        xsfc_conv = self.conv_sfc(xsfc)
        x_conv = ForecastStepBase.combine_pr_sfc(xpr_conv, xsfc_conv, self.config)
        x_conv = self.enc_swin(x_conv)
        return x_conv


class ForecastStepAdapter2(ForecastStepBase, SourceCodeLogger):
    def __init__(self,config):
        super().__init__(config)
        c = self.config ; mesh = c.mesh
        self.conv = nn.Conv3d(in_channels=mesh.n_pr_vars, out_channels=c.conv_dim, kernel_size=c.patch_size, stride=c.patch_size)
        self.conv_sfc = nn.Conv2d(in_channels=c.total_sfc, out_channels=c.conv_dim, kernel_size=c.patch_size[1:], stride=c.patch_size[1:])
        self.enc_swin = BasicLayer3D(dim=c.hidden_dim, input_resolution=c.resolution, depth=c.adapter_swin_depth, num_heads=c.num_heads, window_size=c.window_size, FLASH=c.FLASH, drop_path=c.drop_path, checkpoint_every=c.checkpoint_every, checkpointfn=c.checkpointfn)
        
        assert False, "This is being depricated on Sep 2"
        self.encoder = ForecastStepEncoder(self.config)  
        self.decoder = ForecastStepDecoder(dt=0, config=self.config)

        #self.conv = deepcopy(base_model.conv)
        #self.conv_sfc = deepcopy(base_model.conv_sfc)

    def forward(self,x,dt=0):
        c = self.config
        assert dt == 0
        assert self.config.lat_compress == False
        x = self.encoder(x)
        x = self.enc_swin(x)
        y = self.decoder(x)
        return y
    
class LinearAdapter(ForecastStepBase, SourceCodeLogger):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.mesh.n_vars + 2,config.output_mesh.n_vars)
    
    def forward(self,x,dt=0):
        return self.linear(x)
        
