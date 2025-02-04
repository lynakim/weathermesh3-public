from utils import *

import glob
import io


def save_weights(self,buffer,keep=False):
    model_name = f"model_epoch{self.state.epoch}_iter{self.state.n_iter}_step{self.state.n_step}_loss{self.loss:.3f}.pt"
    savepath = os.path.join(self.log_dir,model_name)
    if keep:
        with open(self.keepsave_path,'a') as f:
            f.write(model_name+'\n')
    print("cleaning")
    clean_saves(self.log_dir)
    print("saving")
    with open(savepath,'wb') as f:
        f.write(buffer.getvalue())
    print("Saved to",savepath)
    print("Save size: %.2fMB" % (os.path.getsize(savepath)/1e6))

def clean_saves(log_dir):
    run = log_dir 
    kp = os.path.join(run,'keepsave.txt')
    if not os.path.exists(kp):
        open(kp,'a').close()
    with open(os.path.join(run,'keepsave.txt')) as f:
        keep = f.readlines()
    keepstr = [x.strip() for x in keep]
    # pattern is actually /model_epoch*_iter*_step*_loss*.pt but leaving out step to keep back-compatible
    saves = glob.glob(run+'/model_epoch*_iter*_loss*.pt')
    if len(saves) < 5:
        return
    # more _step back compatibility
    get_iter = lambda x: int(x.split('_iter')[-1].split('_step')[0]) if '_step' in x else int(x.split('_iter')[-1].split('_loss')[0]) 
    iters = sorted([get_iter(x) for x in saves])
    ls = iters[0]
    to_keep = set([ls])
    for i,iter in enumerate(iters):
        if ls+min(500*(i),50_000) < iter:
            to_keep.add(iter)
            ls = iter
    to_keep.update(iters[-5:])
    sorted(to_keep)
    for s in saves:
        if get_iter(s) in to_keep:
            continue
        if os.path.split(s)[-1] in keepstr:
            continue
        print('removing',s)
        os.remove(s)

def save_model_preforked(self):
    params = sum(p.numel() for p in self.model.parameters())
    print("[save] Number of params: %.2fM" % (params/1e6))
    s = 0 
    for name, param in self.model.named_parameters():
        s+=param.numel()
    print("[save] Total params size", "%.2fMB"%(s/1e6*4))
    buffers = [x[0] for x in self.model.named_buffers()]
    save_dict = {
        'model_state_dict': {k: v for k,v in self.model.state_dict().items()},
        'buffer_checksum': self.get_buffers_checksum(_print=False),
        'conf': self.conf,
        'state': self.state,
        'gradscaler_state_dict': self.scaler.state_dict()
        }
    if self.conf.coupled.hell:
        coupled_buffers = [x[0] for x in self.coupled_model.named_buffers()]
        save_dict['coupled_model_state_dict'] = {k: v for k,v in self.coupled_model.state_dict().items() if k not in coupled_buffers}
    if self.conf.optim != 'shampoo':
        save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
    buffer = io.BytesIO()
    torch.save(save_dict, buffer)
    return buffer

def get_validation_subset(val_date_range,timesteps,rank,ndays=1):
    dh_range = (val_date_range[1] - val_date_range[0]).total_seconds() / (3600)
    val_offset = round((np.modf((1+ np.sqrt(5))/2 * (1+rank))[0] * dh_range))
    d1 = val_date_range[0] + timedelta(days=val_offset // 24)
    val_dates = get_dates((d1,d1+timedelta(days=ndays+max(timesteps)//24)))
    return val_dates


def save_img_with_metadata(path,arr):
    pid = os.fork()
    if pid != 0:
        return
    try:
        h, w = arr.shape
        try: arr = np.roll(arr, w//2, axis=1)
        except: arr = np.roll(arr.cpu().numpy(), w//2, axis=1)
        np.nan_to_num(arr, copy=False, nan=0)
        plt.imsave(path,arr)
    except (KeyboardInterrupt,Exception) as e: 
        print("[Save] Image Save Failed" ,e)
        os._exit(1)
    os._exit(0)

    return
    # Could use below code to lable it with metadata but eh
    dpi=72
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.axis('off')
    ax.imshow(arr)
    ax.text(10, 40, metadata, color="black", fontsize=20,antialiased=False,fontname="monospace")

    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
