import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from gen1.model import ForecastStepBase, xonly
from gen1.utils import load_delta_norm



# Things to maybe try:
# better padding
# handle surface seprarately

class ForecastStepAdapterConv(ForecastStepBase):
    def __init__(self, config):
        super().__init__(config)

        self.H1 = config.adapter_H1
        self.H1_sfc = self.H1*4
        H1 = self.H1
        H1_sfc = self.H1_sfc
        dim_seq = self.config.adapter_dim_seq

        ks3 = (3,3,3)
        pad3 = (1,1,1)
        self.activation = self.config.activation
        #self.activation = nn.LeakyReLU(negative_slope=0.02, inplace=True)
        self.combined_n_pr_vars = sum([mesh.n_pr_vars for mesh in self.config.inputs]) 
        self.combined_total_sfc = sum([mesh.n_sfc_vars for mesh in self.config.inputs]) + self.n_addl_vars + self.n_const_vars
        
        if self.config.adapter_use_input_bias:
            bias = np.load('/fast/consts/bias_gfs_hres_era5.npz')
            gfs_sfc_bias = torch.from_numpy(bias['gfs_sfc']).permute(2,0,1).unsqueeze(0).to(torch.float16)
            gfs_pr_bias = torch.from_numpy(bias['gfs_pr']).permute(2,3,0,1).unsqueeze(0).to(torch.float16)
            self.register_buffer('gfs_sfc_bias', gfs_sfc_bias)
            self.register_buffer('gfs_pr_bias', gfs_pr_bias)
            self.combined_n_pr_vars += gfs_pr_bias.shape[1]
            self.combined_total_sfc += gfs_sfc_bias.shape[1]

        def dub3d_conv(in_c, out_c):
            nonlocal ks3, pad3, self
            """
            In the original paper implementation, the convolution operations were
            not padded but we are padding them here. This is because, we need the 
            output result size to be same as input size.
            """
            conv_op = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=ks3, padding=pad3),
                self.activation,
                nn.Conv3d(out_c, out_c, kernel_size=ks3, padding=pad3),
                self.activation
            )
            return conv_op
        
        self.p1_conv = nn.Sequential(
            nn.Conv3d(self.combined_n_pr_vars, H1, kernel_size=(3,5,5), padding=(1,2,2)),
            self.activation,
            nn.Conv3d(H1, H1, kernel_size=(3,5,5), padding=(1,2,2)),
            self.activation
        )
        self.sfc2p_conv = nn.Sequential(
            nn.Conv2d(self.combined_total_sfc, H1, kernel_size=(5,5), padding=(2,2)),
            self.activation,
            nn.Conv2d(H1, H1, kernel_size=(5,5), padding=(2,2)),
            self.activation
        )
        self.sfc1_conv = nn.Sequential(
            nn.Conv2d(self.combined_total_sfc, H1_sfc, kernel_size=(5,5), padding=(2,2)),
            self.activation,
            nn.Conv2d(H1_sfc, H1_sfc, kernel_size=(5,5), padding=(2,2)),
            self.activation
        )

        self.avg_pool3d = nn.AvgPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.e_conv = dub3d_conv(H1, H1)
        self.down_conv1 = dub3d_conv(H1, dim_seq[0])
        self.down_conv2 = dub3d_conv(dim_seq[0], dim_seq[1])
        #self.down_conv3 = dub3d_conv(dim_seq[1], dim_seq[2])
        
        def make_trans(in_c, out_c):
            return nn.ConvTranspose3d(in_channels=in_c, out_channels=out_c,kernel_size=(2,2,2),stride=(2,2,2))

        #self.up_trans1 = make_trans(dim_seq[2], dim_seq[1])
        #self.up_conv1 = dub3d_conv(dim_seq[2], dim_seq[1])
        self.up_trans2 = make_trans(dim_seq[1], dim_seq[0])
        self.up_conv2 = dub3d_conv(dim_seq[1], dim_seq[0])
        self.up_trans3 = make_trans(dim_seq[0], H1)
        self.up_conv3 = dub3d_conv(H1*2, H1)

        self.out_conv = nn.Sequential(
            nn.Conv3d(H1, 5, kernel_size=(3,5,5), padding=(1,2,2)),
            self.activation,
            nn.Conv3d(5, 5, kernel_size=(3,3,3), padding=(1,1,1)),
        ) 
        self.out_conv_sfc = nn.Sequential(
            nn.Conv2d(H1*4, 4, kernel_size=(5,5), padding=(2,2)),
            self.activation,
            nn.Conv2d(4, 4, kernel_size=(3,3), padding=(1,1)),
        )

        decoder = nn.Module()
        c = self.config
        decoder.register_buffer('delta_norm_matrix',torch.from_numpy(load_delta_norm('gfs2era5', c.output_mesh.n_levels, c.output_mesh)[1]))
        self.decoders = nn.ModuleDict({"0":decoder})

    def to(self, *args, **kwargs):
        print("StepConv to", args, kwargs)
        super().to(*args, **kwargs)
        self.decoders["0"].delta_norm_matrix = self.decoders["0"].delta_norm_matrix.to(*args, **kwargs).detach()
        return self


    def forward(self, xs,dt=0,gimme_deltas=False):
        c = self.config
        assert len(xs) == len(c.inputs) + 1
        t0 = int(xs[-1].item())
        x = xs[0]
        B, Nlat, Nlon, _ = x.shape
        xinp0 = xs[0]

        xxps = [self.breakup_pr_sfc(xx,[torch.tensor(t0)]) for xx in xs[:-1]]

        gfs_pr_bias, gfs_sfc_bias = [],[]
        if self.config.adapter_use_input_bias:
            gfs_pr_bias = [self.gfs_pr_bias]
            gfs_sfc_bias = [self.gfs_sfc_bias]        
        xp = torch.cat([xps[0] for xps in xxps]+gfs_pr_bias, dim=1)
        xs = torch.cat([xps[1][:,:(self.config.inputs[i].n_sfc_vars if i<len(xxps)-1 else xps[1].shape[1])] for i,xps in enumerate(xxps)]+gfs_sfc_bias, dim=1)

        xp = self.p1_conv(xp)
        xs2p = self.sfc2p_conv(xs)
        xp = xp + xs2p.unsqueeze(2)
        xs = self.sfc1_conv(xs)
        xs = xs.view(B, self.H1, 4, Nlat, Nlon)
        x = torch.cat((xp, xs), dim=2)
        del xp, xs, xs2p
        #d1 = self.config.checkpointfn(self.e_conv,x,use_reentrant=False)
        #print("x", x.shape)
        d1 = self.e_conv(x)
        #print("d1", d1.shape)
        #d1= x
        d2 = self.avg_pool3d(d1)
        #print("d2", d2.shape)
        d2 = self.down_conv1(d2)
        #print("d2p", d2.shape)
        d3 = self.avg_pool3d(d2)
        #print("d3", d3.shape)
        d3 = self.down_conv2(d3)
        #print("d3p", d3.shape)
        #d4 = self.avg_pool3d(d3)
        #d4 = self.down_conv3(d4)

        #x = self.up_trans1(d4)
        #x = self.up_conv1(torch.cat([d3, x], 1))
        x = self.up_trans2(d3)
        x = self.up_conv2(torch.cat([d2, x], 1))
        x = self.up_trans3(x)
        x = self.up_conv3(torch.cat([d1, x], 1))
        xp = self.out_conv(x[:, :,:28,...]).flatten(1,2)
        xs = self.out_conv_sfc(x[:, :,28:,...].reshape(B, self.H1_sfc, Nlat, Nlon))
        y = torch.cat((xp, xs), dim=1).permute(0,2,3,1)
        if self.config.output_deltas and not gimme_deltas:
            #print(y.shape, xinp0.shape, self.decoders["0"].delta_norm_matrix.shape)
            #print(y.device, xinp0.device, self.decoders["0"].delta_norm_matrix.device)
            y = xinp0 + y * self.decoders["0"].delta_norm_matrix
        return y 

class ForecastStepAdapterConvGutted(ForecastStepBase):
    def __init__(self, config):
        super().__init__(config)

        H1 = 16
        #dim_seq = [128,256,512]
        #dim_seq = [64,128,256]
        dim_seq = [16,32,64]
        ks3 = (3,3,3)
        pad3 = (1,1,1)
        #self.activation = nn.GELU()
        #self.activation = nn.LeakyReLU(negative_slope=0.02, inplace=True)
        self.activation = nn.GELU()

        def dub3d_conv(in_c, out_c):
            nonlocal ks3, pad3, self
            """
            In the original paper implementation, the convolution operations were
            not padded but we are padding them here. This is because, we need the 
            output result size to be same as input size.
            """
            conv_op = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=ks3, padding=pad3),
                self.activation,
                nn.Conv3d(out_c, out_c, kernel_size=ks3, padding=pad3),
                self.activation
            )
            return conv_op
        
        self.p1_conv = nn.Sequential(
            nn.Conv3d(5, 16, kernel_size=(3,5,5), padding=(1,2,2)),
            self.activation,
            nn.Conv3d(16, 16, kernel_size=(3,5,5), padding=(1,2,2)),
            self.activation
        )
        self.sfc2p_conv = nn.Sequential(
            nn.Conv2d(18, 16, kernel_size=(5,5), padding=(2,2)),
            self.activation,
            nn.Conv2d(16, 16, kernel_size=(5,5), padding=(2,2)),
            self.activation
        )
        self.sfc1_conv = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=(5,5), padding=(2,2)),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=(5,5), padding=(2,2)),
            self.activation
        )

        self.e_conv = dub3d_conv(H1, H1)


        self.out_conv = nn.Sequential(
            nn.Conv3d(16, 5, kernel_size=(3,5,5), padding=(1,2,2)),
            self.activation,
            nn.Conv3d(5, 5, kernel_size=(3,3,3), padding=(1,1,1)),
        ) 
        self.out_conv_sfc = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=(5,5), padding=(2,2)),
            self.activation,
            nn.Conv2d(4, 4, kernel_size=(3,3), padding=(1,1)),
        )
        decoder = nn.Module()
        c = self.config
        decoder.register_buffer('delta_norm_matrix',torch.from_numpy(load_delta_norm('gfs2era5', c.output_mesh.n_levels, c.output_mesh)[1]))
        self.decoders = {"0":decoder}

        #self.linear = nn.Linear(146,144)
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.decoders["0"].delta_norm_matrix = self.decoders["0"].delta_norm_matrix.to(*args, **kwargs).detach()
        return self


    def forward(self, xs, dt=0):
        c = self.config
        
        x,t0 = xonly(xs)

        B, Nlat, Nlon, D = x.shape

        xp, xs = self.breakup_pr_sfc(x,t0)
        if 1:
            xp = self.p1_conv(xp)
            xs2p = self.sfc2p_conv(xs)
            xp = xp + xs2p.unsqueeze(2)
            xs = self.sfc1_conv(xs)
            xs = xs.view(B, 16, 4, Nlat, Nlon)
            x = torch.cat((xp, xs), dim=2)
            x = self.e_conv(x)
            xp = self.out_conv(x[:, :,:28,...]).flatten(1,2)
            xs = self.out_conv_sfc(x[:, :,28:,...].reshape(B, 16*4, Nlat, Nlon))
        else:
            xp = xp.flatten(1,2)
            xs = xs[:,:4,...]
        y = torch.cat((xp, xs), dim=1).permute(0,2,3,1)
        return y 


class ForecastStepAdapterLinear(ForecastStepBase):
    def __init__(self, config):
        super().__init__(config)
        D = config.mesh.n_vars
        self.linear = nn.Linear(D,D)
        decoder = nn.Module()
        c = self.config
        decoder.register_buffer('delta_norm_matrix',torch.from_numpy(load_delta_norm('gfs2era5', c.output_mesh.n_levels, c.output_mesh)[1]))
        self.decoders = {"0":decoder}
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.decoders["0"].delta_norm_matrix = self.decoders["0"].delta_norm_matrix.to(*args, **kwargs).detach()
        return self

    def forward(self, xs, dt=0):
        y = self.linear(xs[0])
        return y 


class ForecastStepAdapterCombo(ForecastStepBase):
    def __init__(self, adapter, forecaster):
        super().__init__(adapter.config)
        self.adapter = adapter
        self.forecaster = forecaster
        for param in self.forecaster.parameters():
            param.requires_grad = False
        try: self.decoders = self.forecaster.decoders
        except: self.decoders = {}
        self.decoders['0'] = adapter.decoders['0']

    def forward(self, xs, dt=0,gimme_deltas=False):
        if dt == 0:
            return self.adapter(xs,dt=dt,gimme_deltas=gimme_deltas)
        else:
            if len(xs) == 3:
                print("using adapter", dt)
                #x = self.adapter(xs,dt=dt)     
                xsp = [a + torch.zeros(1, dtype=a.dtype, device=a.device, requires_grad=True) for a in xs[:-1]] + [xs[-1]]
                x = checkpoint.checkpoint(self.adapter,xsp, use_reentrant=False)
                #x = xsp[0]
                #x = torch.cat((x, torch.zeros(1, 720, 1440, 3, dtype=torch.float16, device=x.device)), axis=3)
                #if self.adapter.config.output_deltas: x = xsp[0] + x * self.adapter.decoders["0"].delta_norm_matrix
                #assert x.requires_grad
                #print("uhhh x requires grad???", x.requires_grad)
                #print("xs[0] requires grad???", xs[0].requires_grad)
                #assert False
                xs = [x, xs[-1]]
                return self.forecaster(xs,dt=dt,gimme_deltas=gimme_deltas),x#[..., :144], x
            else:
                print("not using adapter", dt)
                return self.forecaster(xs,dt=dt,gimme_deltas=gimme_deltas), xs[0]
        
    def to(self, *args, **kwargs):
        print("ForecastStepAdapterCombo.to", args, kwargs)
        super().to(*args, **kwargs)
        self.forecaster = self.forecaster.to(*args, **kwargs)
        self.adapter = self.adapter.to(*args, **kwargs)
        return self
        
