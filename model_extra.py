from utils import *
from data import *
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class ForecastStepConvOnly(nn.Module, SourceCodeLogger):

    def __init__(self, grid=None):
        super().__init__()
        self.grid = grid

        self.input_deltas = False
        self.output_deltas = False
        self.do_sub = True

        assert not self.input_deltas

        self.input_dim = grid.n_pr_vars * grid.n_levels + grid.n_sfc_vars
        self.O = self.input_dim
        self.D = self.input_dim + self.grid.xpos.shape[-1] + 2

        self.total_sfc = grid.n_sfc_vars + 4
        self.total_pr = grid.n_pr_vars * grid.n_levels

        self.patch_size = (2, 4, 4)
        self.conv_dim = 512

        self.encoder = nn.Sequential(
                nn.Linear(self.D, self.conv_dim),
        )

        self.decoder = nn.Sequential(
                nn.Linear(self.conv_dim, self.input_dim)
        )

        self.resolution = (8, 90, 180)
        self.window_size = (2,6,12)

        self.conv = nn.Conv3d(in_channels=grid.n_pr_vars, out_channels=self.conv_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.conv_sfc = nn.Conv2d(in_channels=self.total_sfc, out_channels=self.conv_dim, kernel_size=self.patch_size[1:], stride=self.patch_size[1:])

        self.deconv = nn.ConvTranspose3d(out_channels=grid.n_pr_vars, in_channels=self.conv_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.deconv_sfc = nn.ConvTranspose2d(out_channels=grid.n_sfc_vars, in_channels=self.conv_dim, kernel_size=self.patch_size[1:], stride=self.patch_size[1:])

        return

    def forward(self, x):
        #xpr, xsfc = xtup

        B,Nlat,Nlon,D = x.shape

        assert self.total_pr + self.total_sfc == D
        xpr = x[:,:,:, :self.total_pr]
        xpr = xpr.view(B, Nlat, Nlon, self.grid.n_pr_vars, self.grid.n_levels)

        xpr = xpr.permute(0, 3, 4, 1, 2)
        xpr_conv = self.conv(xpr)
        xpr_conv = xpr_conv.permute(0, 2, 3, 4, 1)


        xsfc = x[:,:,:, self.total_pr:]
        assert xsfc.shape[-1] == self.total_sfc
        xsfc = xsfc.permute(0, 3, 1, 2)
        xsfc_conv = self.conv_sfc(xsfc)
        xsfc_conv = xsfc_conv.permute(0, 2, 3, 1)

        x_conv = torch.cat((xpr_conv, xsfc_conv[:, np.newaxis]), axis=1)
        x_conv = torch.flatten(x_conv, start_dim=1, end_dim=3)

        x_tr = x_conv
        x_tr = x_tr.permute(0, 2, 1)
        x_tr = x_tr.view(B, self.conv_dim, self.resolution[0], self.resolution[1], self.resolution[2])

        y_pr_conv = x_tr[:, :, :-1, :, :]
        y_sfc_conv = x_tr[:, :, -1, :, :]

        y_pr = self.deconv(y_pr_conv)
        y_sfc = self.deconv_sfc(y_sfc_conv)
        y_sfc = y_sfc.permute(0, 2, 3, 1)
        y_pr = y_pr.permute(0, 3, 4, 1, 2)
        y_pr = torch.flatten(y_pr, start_dim=-2)

        y = torch.cat((y_pr, y_sfc), axis=-1)

        return y
    

class ForecastStepRandom(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = 146

    def forward(self, x, dt):
        B,Nlat,Nlon,D = x.shape
        y = torch.randn(B,Nlat,Nlon,self.D)
        return y