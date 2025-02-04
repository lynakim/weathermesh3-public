from model_latlon.top import *

class ForecastCombinedDiffusion(nn.Module,SourceCodeLogger):
    def __init__(self, forecaster, diffuser, T=1000, schedule='linear'):
        super().__init__()
        self.mesh = forecaster.mesh
        self.forecaster = forecaster
        self.diffuser = diffuser
        self.T = T
        self.schedule = schedule
        # Cosine schedule as proposed in the Improved DDPM paper
        # https://arxiv.org/abs/2102.09672
        def cosine_beta_schedule(timesteps, s=0.008):
            """
            cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            """
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)

        # Calculate betas using cosine schedule
        if schedule == 'cosine':
            betas = cosine_beta_schedule(T)
        elif schedule == 'linear':
            beta_start = 1e-4
            beta_end = 0.02
            betas = torch.linspace(beta_start, beta_end, self.T) 
        else:
            assert False, f"Unknown schedule {schedule}"
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        if 1:
            plt.plot(alphas_bar)
            plt.grid()
            plt.savefig(f'imgs/alphas_bar_{schedule}.png')
            plt.close()
            plt.plot(betas)
            plt.grid()
            plt.savefig(f'imgs/betas_{schedule}.png')
            plt.close()
        sqrt_alphas_bar = torch.sqrt(alphas_bar)
        sqrt_m1_alphas_bar = torch.sqrt(1 - alphas_bar)

        # Register buffers
        self.register_buffer('beta', betas)  # β
        self.register_buffer('alpha', alphas)  # α
        self.register_buffer('alpha_bar', alphas_bar)  # ᾱ (cumulative product)
        self.register_buffer('sqrt_alpha_bar', sqrt_alphas_bar)  # √ᾱ
        self.register_buffer('sqrt_m1_alpha_bar', sqrt_m1_alphas_bar)  # √(1-ᾱ)

    def predict(self, x_full, dT):
        with torch.no_grad():
            pdT = self.forecaster.config.processor_dt[0]
            x_clone = [xx.clone() for xx in x_full]
            if type(self.forecaster).__name__ == 'ForecastStep3D':
                for i in range(3):
                    assert not (x_clone[0][:,:,:,-3+i] == 0).all() 
                    x_clone[0][:,:,:,-3+i] = 0 # need to zero out for bachelor 
            todo = {dT : ','.join(['E'] + [f'P{pdT}']*(dT//pdT))}
            latent = self.forecaster(x_clone, todo)[dT]
            c = latent
            if type(self.forecaster).__name__ == 'ForecastStep3D':
                c = c.view(1,*self.forecaster.config.resolution, self.forecaster.config.hidden_dim)
            assert (self.forecaster.config.window_size[2]-1) / 2 == 3.0
            c = c[:,-1,:,3:-3] # -1 is for surface 
            c = c.permute(0,3,1,2) # B,H,W,C -> B,C,H,W
        return latent,c

    def forward(self, x_full, dT, y_t, t, c=None, bbox=None):
        # little t is time in diffusion, big T is time in forecast
        
        # x_t: the sample with the noise added
        # t: the timestep for the diffusion process
        # x_full: the full weather instance for the predictive model
        # dT: forecast hour

        assert x_full[0].shape[1] == 720 and x_full[0].shape[2] == 1440, f"x_full is {x_full[0].shape}"
        
        if c is None:
            latent,c = self.predict(x_full, dT)

        if bbox is not None: 
            csub = get_bbox(bbox, c)
            y_t = get_bbox(bbox, y_t)
        else: csub = c
        
        noise_pred = self.diffuser(y_t, t, csub)
        
        #noise_pred = torch.zeros_like(y_t) 

        todo = {dT : 'D'}

        #y_pred = self.forecaster(latent, todo)[dT]
        #y_pred = None

        return noise_pred, c 
    
    def generate(self, x_full, dT, steps=None):
        if self.schedule == 'cosine':
            return self.generate_cosine(x_full, dT, steps)
        elif self.schedule == 'linear':
            return self.generate_linear(x_full, dT, steps)
        else:
            assert False, f"Unknown schedule {self.schedule}"



    def generate_linear(self, x_full, dT, steps=None):
        assert self.schedule == 'linear'
        _,c = self.predict(x_full, dT)
        device = c.device

        B = 1
        sample = torch.randn(B, self.mesh.n_sfc_vars, len(self.mesh.lats), len(self.mesh.lons), device=device)
        
        # If num_steps is not provided, use the default number of steps
        num_steps = steps or self.T
        
        # Create a new time step sequence
        time_steps = torch.linspace(self.T - 1, 0, num_steps, device=device)
        
        for i in range(num_steps):
            print(f"Generating Step: {i+1}/{num_steps}")
            t = time_steps[i]
            t_prev = time_steps[i+1] if i < num_steps - 1 else torch.tensor(-1, device=device)
            
            # Interpolate alpha values
            alpha = self.alpha_bar[t.long()]
            alpha_prev = self.alpha_bar[t_prev.long()] if i < num_steps - 1 else torch.tensor(1.0, device=device)
            
            t_batch = t.repeat(B)
            noise_pred = self.diffuser(sample, t_batch, c)
            
            # Compute the variance
            variance = (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
            
            # Compute the "direction pointing to x_t"
            pred_original_sample = (sample - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            
            # Compute x_{t-1}
            sample = torch.sqrt(alpha_prev) * pred_original_sample + \
                    torch.sqrt(1 - alpha_prev - variance) * noise_pred + \
                    torch.sqrt(variance) * torch.randn_like(sample)

        sample = sample.permute(0,2,3,1)
        print(sample.shape)
        #edges are fucked rn
        ed = 2
        sample[:, :ed,:,:] = 0; 
        sample[:,-ed:,:,:] = 0; 
        sample[:,:, :ed,:] = 0; 
        sample[:,:,-ed:,:] = 0

        return sample

    def generate_cosine(self, x_full, dT, steps=None):
        assert False, 'this is broken'
        assert self.schedule == 'cosine'
        _,c = self.predict(x_full, dT)
        device = c.device

        B = 1
        sample = torch.randn(B, self.mesh.n_sfc_vars, len(self.mesh.lats), len(self.mesh.lons), device=device)
        
        # If num_steps is not provided, use the default number of steps
        num_steps = steps or self.T
        
        # Create a new time step sequence
        time_steps = torch.linspace(self.T - 1, 0, num_steps, device=device)
        
        # ADDED: Pre-compute betas for numerical stability
        betas_t = torch.zeros(num_steps, device=device)
        for i in range(num_steps):
            t = time_steps[i]
            t_int = int(t.item())
            betas_t[i] = self.beta[t_int]
        
        for i in range(num_steps):
            print(f"Generating Step: {i+1}/{num_steps}")
            t = time_steps[i]
            t_int = int(t.item())  # ADDED: Get integer time step
            
            # REMOVED: t_prev calculation as we'll use a different approach
            
            # CHANGED: Use alpha and alpha_bar directly from schedule
            alpha = self.alpha[t_int]  # Changed from alpha_bar to alpha
            alpha_bar = self.alpha_bar[t_int]
            
            t_batch = t.repeat(B)
            noise_pred = self.diffuser(sample, t_batch, c)
            
            # CHANGED: Simplified variance and mean calculation
            if i < num_steps - 1:
                beta_t = betas_t[i]
                
                # CHANGED: Compute mean with more stable formula
                mean = (1 / torch.sqrt(alpha)) * (
                    sample - (beta_t / torch.sqrt(1 - alpha_bar)) * noise_pred
                )
                
                # CHANGED: Simpler variance calculation
                variance = beta_t
                
                # CHANGED: Simplified posterior sampling
                sample = mean + torch.sqrt(variance) * torch.randn_like(sample)
            else:
                # ADDED: Special handling for final step
                beta_t = betas_t[i]
                sample = (1 / torch.sqrt(alpha)) * (
                    sample - (beta_t / torch.sqrt(1 - alpha_bar)) * noise_pred
                )

        # Keep your original post-processing
        sample = sample.permute(0,2,3,1)
        print(sample.shape)
        #edges are fucked rn
        ed = 2
        sample[:, :ed,:,:] = 0
        sample[:,-ed:,:,:] = 0
        sample[:,:, :ed,:] = 0
        sample[:,:,-ed:,:] = 0

        return sample

