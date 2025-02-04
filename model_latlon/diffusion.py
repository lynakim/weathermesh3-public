from model_latlon.top import *
import tqdm

class ForecastCombinedDiffusion(nn.Module,SourceCodeLogger):
    def __init__(self, forecaster, diffuser, T=1000, schedule='linear', append_input = False, deltas=True):
        super().__init__()
        self.code_gen = 'gen3'
        self.mesh = forecaster.config.outputs[0]
        self.forecaster = forecaster
        self.diffuser = diffuser
        self.T = T
        self.schedule = schedule
        self.append_input = append_input
        self.deltas = deltas
        #assert schedule == 'linear', "cos still broken"
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

        dnorm = f"/fast/consts/diffusion_scaling/stds_{self.forecaster.name}_24.pickle"
        with open(dnorm, "rb") as f:
            norms = pickle.load(f)
            norms_t = torch.zeros(self.mesh.n_sfc_vars,dtype=torch.float16)
            for i,v in enumerate(self.mesh.sfc_vars):
                norms_t[i] = norms[v]
            self.register_buffer('diffusion_norms', norms_t)

    def predict(self, x_full, dTs):
        assert isinstance(dTs, list), f"expects a list of dTs, but got {dTs}"
        with torch.no_grad():
            pdT = self.forecaster.config.processor_dt[0]
            x_clone = [xx.clone() for xx in x_full]
            if self.forecaster.code_gen == 'gen2':
                for i in range(3):
                    assert not (x_clone[0][:,:,:,-3+i] == 0).all() 
                    x_clone[0][:,:,:,-3+i] = 0 # need to zero out for bachelor 

            def mkstr(dT,ending=None):
                st = ','.join(['E'] + [f'P{pdT}']*(dT//pdT) + (ending or []))
                return st
            todo = {dT : mkstr(dT,['D']) for dT in dTs}
            todo.update({f'{dT}L' : mkstr(dT) for dT in dTs})
            preds = self.forecaster(x_clone, todo)
            for k,v in preds.items():
                if type(k) == int:
                    preds[k] = v[:,:,:,-self.mesh.n_sfc_vars:]
            for dT in dTs:
                latent = preds[f'{dT}L']
                c = latent
                if type(self.forecaster).__name__ == 'ForecastStep3D':
                    c = c.view(1,*self.forecaster.config.resolution, self.forecaster.config.hidden_dim)
                assert (self.forecaster.config.window_size[2]-1) / 2 == 3.0
                c = c[:,-1,:,3:-3] # -1 is for surface 
                c = c.permute(0,3,1,2) # B,H,W,C -> B,C,H,W
                preds[f'{dT}C'] = c
        return preds

    def predict_ar(self, x_full, dT):
        if type(self.forecaster).__name__ == 'ForecastStep3D':
            for i in range(3):
                x_full[0][:,:,:,-3+i] = 0 # need to zero out for bachelor 
        out = self.forecaster(x_full, dT)
        return out[dT[0]][:,:,:,-self.mesh.n_sfc_vars:]

    def forward(self, x_full, dT, y_t, t, c=None, bbox=None):
        # little t is time in diffusion, big T is time in forecast
        
        # x_t: the sample with the noise added
        # t: the timestep for the diffusion process
        # x_full: the full weather instance for the predictive model
        # dT: forecast hour

        assert isinstance(dT, int), f"expects just a single dT, but got {dT}"
        assert self.deltas, "only implemented for deltas"

        assert x_full[0].shape[1] == 720 and x_full[0].shape[2] == 1440, f"x_full is {x_full[0].shape}"
        
        if c is None:
            latents,cs = self.predict(x_full, [dT])
            c = cs[dT]

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
    
    def generate(self, x_full, dT, steps=None,num=1,return_ar = False,plot_lines=False):
        #assert self.schedule == 'linear' 
        preds = self.predict(x_full, [dT]) 
        c = preds[f'{dT}C']
        device = c.device

        Bmax = 4
        B = min(Bmax, num)
        assert num < Bmax or num % Bmax == 0, f"num must be a multiple of Bmax, but got {num} and Bmax={Bmax}"
        num_steps = steps or self.T
        time_steps = [t.long() for t in torch.linspace(self.T - 1, 0, num_steps, device=device)]

        samples = []


        if plot_lines:
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        for j in range(num//B):
            sample = torch.randn(B, self.mesh.n_sfc_vars, len(self.mesh.lats), len(self.mesh.lons), dtype=torch.float16, device=device)
            for i in tqdm.tqdm(range(num_steps)):
                #print(f"Generating Step: {i+1}/{num_steps}")
                t = time_steps[i]
                t_prev = time_steps[i+1] if i < num_steps - 1 else torch.tensor(-1, device=device)
                
                # Interpolate alpha values
                alpha = self.alpha_bar[t]
                alpha_prev = self.alpha_bar[t_prev] if i < num_steps - 1 else torch.tensor(1.0, device=device)
                
                t_batch = t.repeat(B)
                if self.append_input:
                    inp = torch.cat([sample, preds[dT].permute(0,3,1,2).repeat(B,1,1,1)], dim=1)
                else:
                    inp = sample
                with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
                    noise_pred = self.diffuser(inp, t_batch, c)
                
                # Compute the variance
                variance = (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
                
                # Compute the "direction pointing to x_t"
                pred_original_sample = (sample - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)

                #plt.imsave(f'ignored/ohp{i:03d}.png',pred_original_sample[0,-3,:,:].cpu().numpy())

                # Compute x_{t-1}
                sample = torch.sqrt(alpha_prev) * pred_original_sample + \
                        torch.sqrt(1 - alpha_prev - variance) * noise_pred + \
                        torch.sqrt(variance) * torch.randn_like(sample)

                #print(f'num nans: {torch.isnan(sample).sum()}, var: {variance} , alpha: {alpha}')
                if plot_lines:
                    ax.plot(pred_original_sample[0,5,600,:].cpu().detach(),alpha=0.5,label=f't={i}')

            if plot_lines:
                ax.legend()
                plt.savefig(f'ignored/lines.png')
                plt.close()

            sample = sample.permute(0,2,3,1)
            #edges are fucked rn
            ed = 2

            if 0:
                for i,v in enumerate(self.mesh.sfc_vars):
                    g = sample[:,ed:-ed,ed:-ed,i].flatten().to(torch.float32)
                    a = ar[:,ed:-ed,ed:-ed,i].flatten().to(torch.float32)
                    xs, xc = torch.linalg.lstsq(torch.stack([g, torch.ones_like(g)], dim=1), a)[0]
                    print(v,xs,xc)
                    if j == 0:
                        sample[:,:,:,i] = sample[:,:,:,i] * xs + xc
            sample[:, :ed,:,:] = 0; 
            sample[:,-ed:,:,:] = 0; 
            sample[:,:, :ed,:] = 0; 
            sample[:,:,-ed:,:] = 0
            if self.deltas:
                sample = sample * self.diffusion_norms + preds[dT]
            sample = sample.cpu().detach()
            for i in range(B):
                samples.append(sample[i].unsqueeze(0))
        if not return_ar:
            return samples
        else:
            return samples, preds[dT]


    def generate_slow(self, x_full, dT, num=1, steps=None,return_ar=False, plot_lines=False):
        # Get predictions
        preds = self.predict(x_full, [dT])
        c = preds[f'{dT}C']
        device = c.device
        
        if plot_lines:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        # Initialize sample with noise
        sample = torch.randn(num, self.mesh.n_sfc_vars, len(self.mesh.lats), len(self.mesh.lons), 
                            dtype=torch.float16, device=device)
        
        # Go through each timestep
        for i in reversed(range(self.T)):
            t_batch = torch.full((num,), i, device=device, dtype=torch.long)
            
            # Prepare input
            if self.append_input:
                inp = torch.cat([sample, preds[dT].permute(0, 3, 1, 2)], dim=1)
            else:
                inp = sample
            
            # Get noise prediction
            with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
                noise_pred = self.diffuser(inp, t_batch, c)
            
            if i > 0:
                beta_t = 1 - self.alpha_bar[i] / self.alpha_bar[i-1]  # Calculate beta_t
                alpha_t = 1 - beta_t  # Calculate alpha_t
                alpha_cumprod_t = self.alpha_bar[i]
                alpha_cumprod_prev = self.alpha_bar[i-1]
                variance = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
                
                # Only add noise if not the last step
                noise = torch.randn_like(sample) if i > 1 else torch.zeros_like(sample)
                
                # Update sample
                sample = (1 / torch.sqrt(alpha_t)) * (
                    sample - ((beta_t / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred)
                ) + torch.sqrt(variance) * noise
            else:
                # Last step
                beta_0 = 1 - self.alpha_bar[0]  # Calculate beta_0
                sample = (1 / torch.sqrt(1 - beta_0)) * (
                    sample - ((beta_0 / torch.sqrt(1 - self.alpha_bar[0])) * noise_pred)
                )
            
            print(f'Step {i}/{self.T}, num nans: {torch.isnan(sample).sum()}')
            
            if plot_lines:
                ax.plot(sample[0, 5, 600, :].cpu().detach(), alpha=0.5, label=f't={i}')
        
        if plot_lines:
            ax.legend()
            plt.savefig('ignored/lines.png')
            plt.close()
        
        # Post-process the samples
        sample = sample.permute(0, 2, 3, 1)
        ed = 2  # edge size to zero out
        
        # Zero out edges
        sample[:, :ed, :, :] = 0
        sample[:, -ed:, :, :] = 0
        sample[:, :, :ed, :] = 0
        sample[:, :, -ed:, :] = 0
        
        # Denormalize
        if self.deltas:
            sample = sample * self.diffusion_norms + preds[dT]
        
        # Prepare output
        sample = sample.cpu().detach()
        samples = [sample[i].unsqueeze(0) for i in range(num)]
        
        return samples