
# dumping diffusion stuff here for now.
class DiffusionMethods:
    def compute_all_diffusion(self,x_gpu, yt_gpus, dts):
        assert len(yt_gpus) == len(dts)
        B = x_gpu[0].shape[0]
        T = self.model.T

        if self.state.n_step % 1000 == 0:# and self.state.n_step != 0:
            with torch.no_grad():
                self.model.eval()
                dt_idx = np.random.randint(0, len(dts))
                ref = yt_gpus[dt_idx][:,:,:,-self.mesh.n_sfc_vars:]
                with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
                    gen, ar = self.model.generate(x_gpu, dts[dt_idx], return_ar=True, steps=50,num=1)
                    gen = gen[0]
                diff = ar - ref
                ref = ref.cpu().detach()
                def to0_1(t):
                    m = t.min(); M = t.max()
                    return (t-m)/(M-m)
                ar = to0_1(ar).cpu().detach()
                diff = to0_1(diff).cpu().detach()
                save_dir = self.log_dir + "/diffusion/"
                os.makedirs(save_dir, exist_ok=True)
                for i,v in enumerate(self.mesh.sfc_vars):
                    print("uhh wtf", ref.shape, gen.shape)
                    img = torch.vstack((ref[0,:,:,i], gen[0,:,:,i]))
                    pb = f'{save_dir}/{self.state.n_step}_{v}_dt={dts[dt_idx]}_{self.rank}'
                    save_img_with_metadata(f"{pb}.png", img)
                    img = torch.vstack((ar[0,:,:,i], diff[0,:,:,i]))
                    save_img_with_metadata(f"{pb}_ar_diff.png", img)
                    plt.title(f'{self.state.n_step} {v} {dts[dt_idx]} lat = {self.mesh.lats[500]:.2f}')
                    plt.plot(gen[0,500,:,i])
                    plt.plot(ref[0,500,:,i])
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig(f"{pb}_plot.png")
                    plt.close()
                self.model.train()



        bbox = self.conf.bboxes[self.state.n_step % len(self.conf.bboxes)]
        dt_idx = (self.state.n_step//len(self.conf.bboxes)) % len(dts)

        print(f"## Doing Forwards {dts[dt_idx]} {bbox}",only_rank_0=True)

        
        if self.num_rerun_sample == 0:
            with torch.no_grad():
                with torch.autocast(enabled=bool(self.conf.HALF), device_type='cuda', dtype=torch.float16):
                    preds = self.model.predict(x_gpu, dts)
                    self.diffusion_c = preds
                    for i in range(len(dts)):
                        for j in range(len(self.mesh.sfc_vars)):
                            assert not (yt_gpus[i][...,-j] == 0).all(), "All zeros in y" 
                        yt_gpus[i] = (yt_gpus[i][:,:,:,-self.model.config.inputs[0].n_sfc_vars:] - preds[dts[i]]).permute(0,3,1,2) / self.model.diffusion_norms.view(1,-1,1,1)
                        yt_gpus[i] = torch.cat([yt_gpus[i], preds[dts[i]].permute(0,3,1,2)], dim=1)


        assert B == 1, "Batch size must be 1 for diffusion"
        t = torch.randint(0, T, (1,), device=self.primary_compute)
        y = yt_gpus[dt_idx]
        with torch.no_grad():
            noise = torch.randn_like(y[:,:self.mesh.n_sfc_vars]) + 0.1 * torch.randn(B,self.mesh.n_sfc_vars,1,1,device=self.primary_compute,dtype=torch.float16)
        alpha_bar_t = self.model.alpha_bar[t]

        # Forward diffusion process
        y_t = y.clone()
        y_t[:,:self.mesh.n_sfc_vars] = (alpha_bar_t**0.5) * y[:,:self.mesh.n_sfc_vars] + (1-alpha_bar_t)**0.5 * noise  # (N, C, H, W)

        with torch.autocast(enabled=bool(self.conf.HALF), device_type='cuda', dtype=torch.float16):
            noise_pred,_ = self.active_model(x_gpu, dts[dt_idx], y_t, t, bbox=bbox, c=self.diffusion_c[f'{dts[dt_idx]}C'])
        
        from model_latlon.top import get_bbox
        noise_sub = get_bbox(bbox, noise)
        loss = F.mse_loss(noise_pred, noise_sub)

        print(f"Diffusion Loss: {loss.item():.4f} t={t.item():d}")

        tbins = [0,30,60,100,200,300,400,600,800,1000]
        for i in range(len(tbins)-1):
            if tbins[i] <= t.item() < tbins[i+1]:
                self.writer.add_scalar(f'LearningDiffusion/loss {tbins[i]}<t<{tbins[i+1]}', loss.item(), self.state.n_step)
                break

        self.Loss[dts[0]] = [loss.item()]
        print("## Doing Backwards",only_rank_0=True)
        self.compute_backward(loss)

    def compute_diffusion_combined(self,x_gpu, yt_gpus, dts):
        assert len(yt_gpus) == len(dts)
        assert len(dts) == 1 
        B = x_gpu[0].shape[0]
        T = self.model.T

        if self.state.n_step % 50 == 0:
            with torch.no_grad():
                self.model.eval()
                ref = yt_gpus[0][:,:,:,-self.mesh.n_sfc_vars:].cpu().detach()
                with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
                    gen = self.model.generate(x_gpu, dts[0], steps=25).cpu().detach()
                save_dir = self.log_dir + "/diffusion/"
                os.makedirs(save_dir, exist_ok=True)
                for i,v in enumerate(self.mesh.sfc_vars):
                    img = torch.vstack((ref[0,:,:,i], gen[0,:,:,i]))
                    save_img_with_metadata(f"{save_dir}/{v}_{self.rank}_{self.state.n_step}.png", img)

                self.model.train()

        print("## Doing Forwards",only_rank_0=True)

        #for name, param in self.active_model.named_parameters():
        #    print(f"{name}: {param.dtype}")


        y = yt_gpus[0][:,:,:,-self.model.config.inputs[0].n_sfc_vars:].clone()
        C = y.shape[3] 
        for i in range(C):
            assert not (y[...,i] == 0).all(), "All zeros in y" 
        y = y.permute(0,3,1,2)
        t = torch.randint(0, T, (B,), device=self.primary_compute).long()  # (N,)

        noise = torch.randn_like(y) 
        sqrt_alpha_cumprod_t = self.model.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)  # (N, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # Forward diffusion process
        y_t = sqrt_alpha_cumprod_t * y + sqrt_one_minus_alpha_cumprod_t * noise  # (N, C, H, W)

        with torch.autocast(enabled=bool(self.conf.HALF), device_type='cuda', dtype=torch.float16):
            noise_pred,y_gpu = self.active_model(x_gpu, dts[0], y_t, t)
        loss = F.mse_loss(noise_pred, noise)
        print("loss diffusion", loss.item())

        loss_ar = torch.nansum(self.weight * (torch.abs(y_gpu - yt_gpus[0]))) / self.sum_weight / B
        print("loss autoregressive", loss_ar.item())

        loss += loss_ar

        self.Loss[dts[0]] = [loss.item()]

        self.computeErrors(x_gpu, y_gpu ,yt_gpus[0],dts[0])


        print("## Doing Backwards",only_rank_0=True)
        self.compute_backward(loss)
