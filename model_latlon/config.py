from meshes import LatLonGrid

from model_latlon.rotary import (
    RotaryEmbedding,
    Liere
)

class ForecastModelConfig():
    def __init__(self,inputs,**kwargs):
        self.inputs = inputs
        self.outputs = inputs
        self.harebrained = False
        self.processor_dts = [6]
        self.parallel_encoders = False
        self.encoder_weights = [1,1,1,1,1,1]
        self.window_size = (5,7,7)
        self.oldpr = False
        self.pr_dims = [32,64,256]
        self.encdec_tr_depth = 2
        self.latent_size = 1024
        self.dims_per_head = 32
        self.affine = True
        self.tr_embedding = 'sincos' # or 'rope', or 'liere'
        self.pr_depth = [8]
        self.checkpoint_type = "matepoint" # or "torch" or "none"
        self.nsight = False # used for profiling
        self.patch_size = (4,8,8)
        self.patch_size_9km = (4,20,20)
        self.simple_decoder_patch_size = None
        self.use_pole_convs = True
        self.da_depth = 4
        self.da_perturber = 0
        self.patch_deltas = 0
        
        self.dec_sublatent = 96
        self.deconv_mlp_dim = 512
        self.deeper_m2g = False

        self.n_chunks_dec = 10

        self.ens_nomean = False

        self.cross_entropy_weight = 0.03 # yeahhh this should really be in trainer config
        # for context: 1 is way too big. 0.1 is more reasonable but still too big. 0.03 is still honestly a bit big but whatever

        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a ForecastStepConfig2 attribute"
            setattr(self,k,v)
        self.update()

    def update(self): 
        self.num_heads = self.latent_size // self.dims_per_head

        omesh = self.outputs[0]
        assert isinstance(omesh, LatLonGrid)
        self.latent_lats = omesh.lats[::self.patch_size[1]]
        self.latent_lons = omesh.lons[::self.patch_size[2]]
        assert self.patch_size[1] == self.patch_size[2], "patch_size must be square"
        if self.simple_decoder_patch_size is None:
            self.simple_decoder_patch_size = self.patch_size
        self.latent_res = omesh.res * self.patch_size[1]
        self.latent_levels = omesh.levels[::-self.patch_size[0]][::-1]

        #legacy shit
        self.processor_dt = self.processor_dts

        self.embedding_module = None
        if self.tr_embedding == 'rotary':
            self.embedding_module = RotaryEmbedding(
                dim = (self.latent_size//self.num_heads)//3,
                freqs_for = 'pixel',
                max_freq = 256
            )
        elif self.tr_embedding == 'liere':
            self.embedding_module = Liere((self.latent_size//self.num_heads), num_dim=3)
        else:
            assert self.tr_embedding == 'sincos'
