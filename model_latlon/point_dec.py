from utils import *
from model_latlon.primatives2d import *
from model_latlon.decoder import Decoder
from model_latlon.transformer3d import SlideLayers3D, tr_unpad
from model_latlon.transformer2d import SlideLayers2D
from model_latlon.primatives2d import call_checkpointed
from eval import eval_rms_train, eval_rms_point
from model_latlon.transformer3d import add_posemb
from model_latlon.cross_attn import CrossTransformer

POINT_DECODERS = ["PointDecoder", "QueryPointDecoder", "JohnPointDecoder"]

class PointDecoderCommon(Decoder):

    def __init__(self,mesh,config):
        super(PointDecoderCommon, self).__init__(self.__class__.__name__)        
        self.mesh = mesh
        self.config = config
        self.point_vars = mesh.vars
        self.n_point_vars = len(self.point_vars)
        self.variable_loss_weights = [1., 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.5]
        assert len(self.variable_loss_weights) == self.n_point_vars, f"need to update variable_loss_weights in PointDecoder class if changing n_point_vars"

        # Gather loss weights
        # TODO: maybe there should be a variable loss weight for point vars
        self.decoder_loss_weight = self.default_decoder_loss_weight        

        self.load_statics()
        self.setup_norm()

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
        baselat = torch.round(lat + self.local_patch_size / 2).to(torch.int64)
        baselon = torch.round(lon - self.local_patch_size / 2).to(torch.int64)
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
            pad_sizes = [(pad_bottom, pad_top)] + [(0, 0)] * (len(sliced_arr.shape) - 1)
            sliced_arr = torch.nn.functional.pad(
                torch.tensor(sliced_arr),
                [item for pair in reversed(pad_sizes) for item in pair],
                mode='constant',
                value=0
            )
        else:
            sliced_arr = torch.tensor(full_arr[idxlat:idxlat_end])

        # Ensure longitude slicing handles wrapping correctly
        if idxlon_end > idxlon:
            local_arr = sliced_arr[:, idxlon:idxlon_end].clone()
        else:
            # Handle wrapping by concatenating the end and beginning of the array
            local_arr = torch.cat(
                (sliced_arr[:, idxlon:], sliced_arr[:, :idxlon_end]),
                dim=1
            )
        assert local_arr.shape[:2] == (self.local_patch_size*ires, self.local_patch_size*ires), f"local_arr shape: {local_arr.shape}, baselat: {baselat}, baselon: {baselon}, idxlat: {idxlat}, idxlon: {idxlon}, ires: {ires}"
        return local_arr

    def get_statics(self, timestamp, points):
        # Note: This needs to be vectorized, it's v slow in a list
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
                        local_arr * 0 + torch.mean(local_arr) * 0.001,
                        local_arr * 0 + torch.std(local_arr) * 0.005,
                        (local_arr - torch.mean(local_arr)) / (1 + torch.std(local_arr)),
                    ]
                    local_arr = torch.stack(local_elevs, axis=-1).to(torch.float16)
                elif static_type == "Rad":
                    local_arr = ((local_arr - 300)/400).to(torch.float16)[:,:,None]
                elif static_type == "Ang":
                    local_arr = local_arr * torch.pi / 180
                    local_arr = torch.stack([torch.cos(local_arr), torch.sin(local_arr)], axis=-1).to(torch.float16)
                elif static_type == "Static_sfc":
                    local_arr = local_arr.to(torch.float16)
                static_data[static_type].append(torch.tensor(local_arr))
        for key in static_data:
            static_data[key] = torch.stack(static_data[key], axis=0).to('cuda')
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
        lati = torch.clip(((90. - latlons[:, 0]) // res).to(int), 0, H - 1)
        loni = (((180 // res) + (latlons[:, 1] + 180.) // res) % (360 // res)).to(int)
        
        # Create arrays for the 3x3 offsets
        di = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        dj = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        
        # Compute indices for each point in the 3x3 patch
        patch_lati = torch.clip(lati[:, None] + di[None, :], 0, H - 1)  # Clip latitude at poles
        patch_loni = (loni[:, None] + dj[None, :]) % W  # Wrap longitude around globe
        
        # Index into the latent space
        x = x[0, -1]  # Select batch and time step
        patches = x[patch_lati, patch_loni]  # Shape: (N, 9, hidden_dim)
        
        # Reshape to (N, 3, 3, hidden_dim)
        N = latlons.shape[0]
        patches = patches.reshape(N, 3, 3, -1)
        
        return patches
    
    def index_3d_latent_patch(self, x, latlons):
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
        lati = torch.clip(((90. - latlons[:, 0]) // res).to(int), 0, H - 1)
        loni = (((180 // res) + (latlons[:, 1] + 180.) // res) % (360 // res)).to(int)
        
        # Create arrays for the 3x3 offsets
        di = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=x.device)
        dj = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=x.device)
        
        # Compute indices for each point in the 3x3 patch
        patch_lati = torch.clip(lati[:, None] + di[None, :], 0, H - 1)  # Clip latitude at poles
        patch_loni = (loni[:, None] + dj[None, :]) % W  # Wrap longitude around globe
        
        # Index into the latent space
        patches = x[0,:,patch_lati, patch_loni]  # Shape: (N, D, 9, hidden_dim)
        
        # Reshape to (N, 3, 3, hidden_dim)
        N = latlons.shape[0]
        patches = patches.reshape(N, D, 3, 3, -1)
        
        return patches
    

    def patch_to_point(self, x, latlons, res, extent):
        """
        x: shape (batch, hidden_dim, n, n) where n is 3 * patch_size
        """
        patch_size = int(extent / res)
        idx_in_coarse_patch = np.floor((latlons % extent) / res).astype(int)
        idx_in_finer_patch = idx_in_coarse_patch + np.array([patch_size, patch_size])
        baselat = torch.round(latlons[:, 0] + self.local_patch_size / 2).to(int)
        baselon = torch.round(latlons[:, 1] - self.local_patch_size / 2).to(int)
        offset_lat = torch.floor((baselat - latlons[:, 0]) / res).to(int)
        offset_lon = torch.floor((latlons[:, 1] - baselon) / res).to(int)
        lati, loni = idx_in_finer_patch[:, 0], idx_in_finer_patch[:, 1]
        lati -= offset_lat
        loni -= offset_lon
        return torch.stack([x[i, :, lat:lat+patch_size, lon:lon+patch_size] for i, (lat, lon) in enumerate(zip(lati, loni))], dim=0)

    def downsize_patch(self, x, n, n_new):
        """
        Take the center n_new x n_new patch of the n x n patch
        """
        assert n_new <= n, "downsize_patch: new size must be smaller"
        assert n % n_new == 0, "downsize_patch: new size must be a multiple of old size"
        start = (n - n_new) // 2
        return x[:,:,start:start+n_new,start:start+n_new]


    def compute_loss(self, y_gpu, yt_gpu, station_data):
        #yt_gpu, point_weights = yt_gpu[:self.CAP], point_weights[:self.CAP]
        y_gpu = y_gpu.squeeze() # TODO I don't actually understand why this needs squeezing too
        yt_gpu = yt_gpu.squeeze() # TODO squeeze is there because collate_fn
        #variable_weights = torch.ones(self.n_point_vars, device=y_gpu.device)
        variable_weights = torch.tensor(self.variable_loss_weights, device=y_gpu.device)
        pw = torch.concat([p['weights'] for p in station_data])
        weights = pw[:,None] * variable_weights
        self.loss = F.mse_loss(y_gpu * weights, yt_gpu * weights, reduction='none').mean(axis=0)
        return self.loss.sum()
    
    def compute_errors(self, y_gpu, yt_gpu, trainer):
        #yt_gpu, point_weights = yt_gpu[:self.CAP], point_weights[:self.CAP]
        station_data = yt_gpu
        yt_gpu = torch.concat([p['data'] for p in station_data])

        ddp_reduce = trainer.DDP and not trainer.data.config.random_timestep_subset
        with torch.no_grad():
            point_weights = torch.concat([p['weights'] for p in station_data])
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

class QueryPointDecoder(PointDecoderCommon):
    def __init__(
            self, 
            mesh, 
            config, 
        ):
        super(QueryPointDecoder, self).__init__(mesh, config)

        self.point_vars = self.mesh.vars
        self.n_point_vars = len(self.point_vars)
        
        self.loc_emd_dim = 256
        self.loc_to_query = nn.Linear(self.loc_emd_dim,self.config.latent_size * self.n_point_vars)



    def forward(self, x, timestamp, points):
        """
        x: latent space tensor of shape (B, D, H, W, C)
        timestamp: timestamp of the current data
        points: tensor of shape (B, 2) of latlons
        """
        _,D,H,W,C = x.shape
        x = tr_unpad(x, self.config.window_size) # shape: (1, D, H, W, C)
        x = self.index_latent_patch(x, points) # shape: (B, latent_size, 3, 3)
        points = torch.from_numpy(points).to(x.device)
        lati,loni,lat_th,lon_th = get_lat_lon_th(points[:,0],points[:,1], 2.0)
        loc_emb = sincos_spacial_embed(torch.stack([lat_th,lon_th],dim=-1).to(x.dtype),embed_dim=self.loc_emd_dim // 2)
        query = self.loc_to_query(loc_emb).view(-1,C,self.n_point_vars)
        out = (x[:,1,1,:,None] * query / C** 0.5).sum(dim=1)
        return out          
    
class JohnPointDecoder(PointDecoderCommon):
    def __init__(self, mesh, config):
        super(JohnPointDecoder, self).__init__(mesh, config)

        self.local_patch_size = 1 
        self.point_vars = self.mesh.vars
        self.n_point_vars = len(self.point_vars)
        
        self.hires_region_size_deg = 1.0

        self.hires_vertical_dim = 32
        self.hires_horizontal_dim = 60
        self.hires_hidden_dim = 256
        self.output_conv_upscale = 3

        D,H,C = self.hires_vertical_dim, self.hires_horizontal_dim, self.hires_hidden_dim

        self.latent_proj = nn.Linear(self.config.latent_size, self.hires_hidden_dim*4)

        # the theta embedding tells the model where the center of the high rest grid is relative to the section of the eath latent sapce that it has.
        # it ulimately is added to the earth latent space 
        self.theta_emb_dim = 128
        self.theta_emb_proj = nn.Linear(self.theta_emb_dim, self.hires_hidden_dim)


        assert H % 30 == 0, "hires_horizontal_dim must be divisible by 30"
        assert H % 4 == 0, "hires_horizontal_dim must be divisible by 4"
        self.arcsec30_statics_to_hires = nn.Conv2d(5, C*D, kernel_size= H // 30, stride=H // 30)
        self.p25deg_statics_to_hires = nn.ConvTranspose2d(10, C*D, 
                                                            kernel_size=H // 4, stride=H // 4)
        
        self.cross_tr = CrossTransformer(
            dim = C,
            depth = 2,
            dim_head = 32
        )

        self.tr3d = SlideLayers3D(
            dim= C, 
            depth=2,
            num_heads= C // 32, 
            window_size= (15,21,21), 
            checkpoint_type=self.config.checkpoint_type
        )

        self.sfc_deconv  = nn.ConvTranspose2d(self.hires_hidden_dim,self.n_point_vars*2,kernel_size=self.output_conv_upscale, stride=self.output_conv_upscale) 
        self.sfc_resconv = ResBlock2d(self.n_point_vars*2,self.n_point_vars)

        offsets = torch.linspace(-self.hires_region_size_deg / 2, 
                                 self.hires_region_size_deg / 2, 
                                 self.hires_horizontal_dim * self.output_conv_upscale )

        output_meshgrid = torch.stack(torch.meshgrid(offsets, offsets), dim=-1)
        self.register_buffer('output_meshgrid', output_meshgrid)




    @TIMEIT(sync=True)
    def forward(self, x_earth, station_data, timestamp):
        """
        x: latent space tensor of shape (B, D, H, W, C)
        timestamp: timestamp of the current data
        points: tensor of shape (B, 2) of latlons
        """
        _,D,H,W,C = x_earth.shape
        B = len(station_data)
        print(f"PointDec Batch: {B}")

        station_data = to_device(station_data, x_earth.device)
        centers = torch.stack([s['center'] for s in station_data])

        x_earth = tr_unpad(x_earth, self.config.window_size) # shape: (1, D, H, W, C)
        x_earth = self.index_3d_latent_patch(x_earth, centers) # shape: (B, D, 3, 3 C)
        x_earth = self.latent_proj(x_earth).reshape(B,-1,self.hires_hidden_dim) # B, D*3*3, C just flatten down to 1d list of tokens that for cross attention


        _,_,lat_th,lon_th = get_lat_lon_th(centers[:,0],centers[:,1], 2.0)
        theta_emb = sincos_spacial_embed(torch.stack([lat_th,lon_th],dim=-1).to(x_earth.dtype),embed_dim=self.theta_emb_dim // 2)
        theta_emb = self.theta_emb_proj(theta_emb).view(B,1,self.hires_hidden_dim)

        x_earth += theta_emb

        Dr, Hr = self.hires_vertical_dim, self.hires_horizontal_dim
        Cr = self.hires_hidden_dim

        # patch of D,3,3 becomes Dr, 3*Nr , 3*Nr
        x_hires = torch.zeros(B,Dr,Hr,Hr,Cr, device=x_earth.device, dtype=torch.float16)
        
        x_hires = add_posemb(x_hires)

        # Conv in statics data
        statics = self.get_statics(timestamp, centers)
        st = torch.concat([statics['Elevation30'], statics['Modis30'][:,:,:,None]], dim=-1).permute(0,3,1,2) # B, C, H, W
        st = self.arcsec30_statics_to_hires(st).view(B,Dr,Cr,Hr,Hr)
        st = st.permute(0,1,3,4,2) # B, D, H, W, C
        x_hires += st

        st = torch.concat([statics['Rad'], statics['Ang'], statics['Static_sfc']],dim=-1).permute(0,3,1,2) # B, C, H, W
        st = self.p25deg_statics_to_hires(st).view(B,Dr,Cr,Hr,Hr)
        st = st.permute(0,1,3,4,2) # B, D, H, W, C
        x_hires += st 
        del st

        def ch1(x_hires,x_earth):
            Bc = x_hires.shape[0]
            x_hires = x_hires.view(Bc,Dr*Hr*Hr,Cr)
            x_hires = self.cross_tr(x_hires, x_earth)
            x_hires = x_hires.view(Bc,Dr,Hr,Hr,Cr)
            return x_hires
        
        x_hires = call_checkpointed(ch1,x_hires,x_earth,checkpoint_type=self.config.checkpoint_type)

        def ch2(x_hres):
            # Run 3d Tranformer
            x_hres = self.tr3d(x_hres)
            return x_hres

        x_hires = call_checkpointed(ch2,x_hires,checkpoint_type=self.config.checkpoint_type)

        def ch3(x_hires):
            # Decode to high res grid with statcis added direct
            sfc = self.sfc_deconv(x_hires[:,-1,:,:,:].permute(0,3,1,2))
            sfc = self.sfc_resconv(sfc).permute(0,2,3,1)
            return sfc

        sfc = call_checkpointed(ch3,x_hires,checkpoint_type=self.config.checkpoint_type)

        B,Co,Ho,Ho = sfc.shape

        def find_nearest_neighbors(points, meshgrid):
            points = points.view(points.shape[0], 1, 1, 2)
            distances = ((points - meshgrid.unsqueeze(0)) ** 2).sum(dim=-1)
            min_indices = torch.min(distances.view(points.shape[0], -1), dim=1)[1]
            row_indices = min_indices // meshgrid.shape[1]
            col_indices = min_indices % meshgrid.shape[1]
            return torch.stack([row_indices, col_indices], dim=1)

        out = []
        for i,region in enumerate(station_data):
            locs = region['latlons'] - centers[i]
            indices = find_nearest_neighbors(locs, self.output_meshgrid)
            out.append(sfc[i,indices[:,0],indices[:,1],:])
        sout = torch.concat(out, dim=0)
        return sout
