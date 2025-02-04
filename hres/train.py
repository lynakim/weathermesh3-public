from data import *
import sys
import glob
import socket
import io
import pickle
import traceback
import numpy as np
import os
import torch
#torch.manual_seed(0)
from hres.hres_utils import *

from hres.model import HresModel

from torch.utils.data.sampler import WeightedRandomSampler

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.utils.shampoo_utils import GraftingType

import torch.utils.checkpoint as checkpoint
import torch.distributed.checkpoint as dist_checkpoint

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch.utils.tensorboard import SummaryWriter

station_list = pickle.load(open("/fast/ignored/hres/station_list.pickle", "rb"))

def am_i_torchrun():
    return 'TORCHELASTIC_ERROR_FILE' in os.environ.keys()

# i swear to god john
def save_weights(self,buffer,keep=False):
    model_name = f"model_step{self.step}_loss{self.avg_loss:.3f}.pt"
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
    saves = glob.glob(run+'/model_step*_loss*.pt')
    if len(saves) < 5:
        return
    get_iter = lambda x: int(x.split('_step')[1].split('_loss')[0])
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
    #    'buffer_checksum': self.get_buffers_checksum(_print=False),
    save_dict = {
        'model_state_dict': {k: v for k,v in self.model.state_dict().items() if k not in buffers},
        'loss': self.avg_loss,
        'conf': self.conf,
        'step': self.step,
        'gradscaler_state_dict': self.scaler.state_dict()
        }
    if self.conf.optim != 'shampoo':
        save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
    buffer = io.BytesIO()
    torch.save(save_dict, buffer)
    return buffer
 

RUNS_PATH = "/fast/ignored"
RUNS_PATH = "/huge/deep"

class NoOp:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class Trainer:
    def __init__(self, conf, dataset, model, simple=False):
        self.last_pid = None
        self.writer = NoOp()
        self.error_dict = {}
        self.B = 512
        self.variables = dataset.variables
        self.dataset = dataset

        self.conf = conf
        self.model = model

        if simple: return

        self.DDP = am_i_torchrun()
        if self.DDP:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        gpus = ["cuda:"+str(i) for r in self.conf.gpus.split(',') for i in range(int(r.split('-')[0]), int(r.split('-')[-1]) + 1)]
        if self.DDP:
            if len(gpus) == 1:
                self.gpus = ["cuda:"+str(self.local_rank)]
            else:
                self.gpus = [gpus[self.local_rank]]
        else:
            self.gpus = gpus
        self.gpus = self.gpus[:torch.cuda.device_count()]
        self.num_gpus = len(self.gpus)

        print("number of parameters", sum([x.numel() for x in self.model.parameters()])/1e6, "M")
        #self.primary_compute = torch.device("cuda")
        #self.model = self.model.to(self.primary_compute)
        #if HALF:
        #    self.model = self.model.half()

        self.scaler = torch.cuda.amp.GradScaler(init_scale=self.conf.initial_gradscale)
        self.scaler.set_growth_interval(1_000)


    def setup_optimizer(self):

        if self.conf.optim == 'adam':
            self.optimizer = torch.optim.AdamW(self.active_model.parameters(), **vars(self.conf.adam))
        else:
            self.optimizer = DistributedShampoo(
                    [x for x in self.active_model.parameters() if x.requires_grad],
                    lr=1e-2,
                    betas=self.conf.adam.betas,
                    epsilon=1e-12,
                    weight_decay=1e-6,
                    max_preconditioner_dim=4096,#8192,
                    precondition_frequency=1, # you probably want to start with this at like, 75!
                    start_preconditioning_step=1,
                    use_decoupled_weight_decay=True,
                    grafting_type=GraftingType.ADAM,
                    grafting_epsilon=1e-08,
                    grafting_beta2=0.999,
                    num_trainers_per_group=10,
                )
    
    def setup_data(self):
        c = self.conf
        self.rank = 0
        self.world_size = 1
        self.primary_compute = torch.device(self.gpus[0]) 
        if self.DDP:
            dist.init_process_group('nccl')
            assert len(self.gpus) == 1, "DDP only works with one GPU per process"
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.primary_compute = torch.device(self.gpus[0])
            torch.cuda.set_device(self.primary_compute)
            self.num_gpus = self.world_size
            print("Starting DDP with rank",self.rank, "world size", self.world_size)
        #elif self.conf.cpu_only:
        #    self.primary_compute = torch.device('cpu')

        torch.manual_seed(time.time()*self.rank + self.rank + 42./(self.rank+1))

        sampler = WeightedRandomSampler(np.array(self.dataset.weights)**(1/2.), num_samples=len(self.dataset.weights), replacement=True)
        self.loader = DataLoader(self.dataset, batch_size=1, num_workers=1 if socket.gethostname() in ["stinson", "singing"] else 2, sampler=sampler)
        self.len_dataset = len(self.dataset)
        
        if '_' in self.conf.name: print("WARNING: underscore in name messes with things, replacing with dash")
        self.conf.name = self.conf.name.replace('_','-')

        self.log_dir = (self.conf.nope*"/tmp/")+f"{RUNS_PATH}/runs_hres/run_{self.conf.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"  
        if self.DDP:
            brdcast = [self.log_dir if self.rank == 0 else None]
            dist.broadcast_object_list(brdcast, src=0)
            self.log_dir = brdcast[0]


        self.model = self.model.to(self.primary_compute)

        self.load_statics()

        if self.DDP:
            self.active_model = DistributedDataParallel(self.model,device_ids=[self.primary_compute])#,find_unused_parameters=True)
        else:
            self.active_model = self.model

        if self.rank != 0: return

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def load_statics(self):
        interps, idxs = pickle.load(open("/fast/ignored/hres/interps%s.pickle"%self.model.grid, "rb"))
        statics = pickle.load(open("/fast/ignored/hres/statics%s.pickle"%self.model.grid, "rb"))
        self.nnew = interps.shape[0]

        interps2, idxs2 = pickle.load(open("/fast/ignored/hres/interps%s_old.pickle"%self.model.grid, "rb"))
        self.nold = interps2.shape[0]
        statics2 = pickle.load(open("/fast/ignored/hres/statics%s_old.pickle"%self.model.grid, "rb"))

        interps = np.concatenate([interps, interps2], axis=0)
        idxs = np.concatenate([idxs, idxs2], axis=0)
        statics = {x: np.concatenate([statics[x], statics2[x]], axis=0) for x in statics}

        self.interps = torch.tensor(interps)#.to(self.primary_compute)
        self.idxs = torch.tensor(idxs)#.to(self.primary_compute)
        self.statics = {x: torch.tensor(y) for x, y in statics.items()} # .to(self.primary_compute)

    def LRsched(self,step):
        c = self.conf.lr_sched
        lr = self.computeLR(step,c)
        self.current_lr = lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr
    
    @staticmethod
    def computeLR(step,c):
        # I'm sorry this is gross but at least it's easy to test with 
        # pytest test_all.py::test_lr

        cycle = c.warmup_end_step + c.cosine_period
        step = max(step + c.step_offset,0)
        n = step // cycle
        step_modc = step % cycle
        lr = np.interp(step+1, [0, c.warmup_end_step], [0, c.lr])
        if step > c.warmup_end_step:
            if c.cosine_en:
                lr = np.interp(step_modc+1, [0, c.warmup_end_step], [0, c.lr / (c.div_factor**n)])
                cstep = step_modc - c.warmup_end_step
                lr = lr * (np.cos(cstep/c.cosine_period *np.pi)+1)/2
                #lr = lr / c.div_factor**(n)
                if c.cosine_bottom is not None:
                    if n > 0:
                        lr = c.cosine_bottom
                    else:
                        lr = max(lr, c.cosine_bottom)
            else: 
                lr = c.lr
        #print("lr", lr, "iter", iter, "step'", step, "n", n,only_rank_0=True)
        return lr


    def setup(self):
        self.neocache = {}
        self.setup_data()
        self.setup_optimizer()
        self.LRsched(0)

    def interpget(self, src, toy, hr, a=300, b=400):
        if src not in self.neocache:
            self.neocache[src] = {}

        def load(xx, hr):
            if (xx,hr) in self.neocache[src]:
                return self.neocache[src][(xx, hr)]
            #print("loading", src, xx, hr)
            if self.conf.HALF:
                f = torch.HalfTensor
            else:
                f = torch.FloatTensor
            ohp = f(((np.load("/fast/consts/"+'/%s/%d_%d.npy' % (src, xx, hr)) - a) / b))
            self.neocache[src][(xx,hr)] = ohp
            return ohp
        avg = load(toy, hr)
        return avg


    def make_sample(self, era5, sample_stations, date, ret_center=False):
        date = date[0]
        date = datetime(1970,1,1)+timedelta(seconds=int(date))
        if type(era5) == dict: assert era5["sfc"].shape[0] == 1

        interp = self.interps[sample_stations]
        idx = self.idxs[sample_stations]

        if self.model.do_pressure:
            if type(era5) == dict:
                pr = era5["pr"]#.to(self.primary_compute)
            else:
                if era5.shape[-1] >= 140:
                    pr = era5[:, :, :, :5*28].view(1, 720, 1440, 5, 28)#.to(self.primary_compute)
                else:
                    pr = era5[:, :, :, :5*25].view(1, 720, 1440, 5, 25)#.to(self.primary_compute)
            pr_sample = pr[0, idx[..., 0], idx[..., 1], :, :]
            pr_sample = torch.sum(pr_sample * interp[:,:,:,:,None,None], axis=3)[0].to(self.primary_compute)

        if self.model.do_radiation:
            soy = date.replace(month=1, day=1)
            toy = int((date - soy).total_seconds()/86400)
            if toy % 3 != 0:
                toy -= toy % 3
            rad = self.interpget("neoradiation_1", toy, date.hour)
            ang = self.interpget("solarangle_1", toy, date.hour, a=0, b=180/np.pi)
            sa = torch.sin(ang)
            ca = torch.cos(ang)
            arrs = [rad, sa, ca]

            extra = []
            for a in arrs:
                #a = a.cuda()
                exa = a[idx[..., 0], idx[..., 1]]
                exa = torch.sum(exa * interp, axis=3)[0]
                del a
                extra.append(exa[:,:,None])

        else:
            extra = []

        if type(era5) == dict: sfc = era5["sfc"]#.to(self.primary_compute)
        else:
            if era5.shape[-1] >= 140:
                sfc = era5[:, :, :, 5*28:5*28+4+self.model.sfc_extra].view(1, 720, 1440, 4)
            else:
                sfc = era5[:, :, :, 5*25:5*25+4+self.model_sfc_extra].view(1, 720, 1440, 4 + self.model.sfc_extra)
        sfc_sample = sfc[0, idx[..., 0], idx[..., 1], :]
        sfc_sample = torch.sum(sfc_sample * interp[:,:,:,:,None], axis=3)[0]
        sera5 = self.statics["era5"][sample_stations][0]
        """ now done inprodcessing btw
        sera5[:, :, 0] /= 1000. * 9.8
        sera5[:, :, 1] /= 7.
        sera5[:, :, 3] /= 20.
        sera5[:, :, 4] /= 20.
        """
        sfc_sample = torch.cat([sfc_sample, sera5] + extra, axis=-1).to(self.primary_compute)

        if self.model.do_pressure:
            center_pr = pr_sample[:,0]
            pr_sample = pr_sample[:, 1:]

        center_sfc = sfc_sample[:,0]

        sfc_sample = sfc_sample[:, 1:]
        sq = int(np.sqrt(sfc_sample.shape[1]))
        #print("hey uh", sfc_sample.shape, sfc_sample.permute(0,2,1).shape)
        sfc_sample = sfc_sample.permute(0, 2, 1).view(-1, sfc_sample.shape[2], sq, sq)

        #static_keys = sorted(statics.keys())
        #static_keys.remove("era5")
        static_keys = ["mn30", "mn75"]
        modis_keys = ["modis_"+x for x in static_keys]
        static = {x: self.statics[x][sample_stations][0] for x in static_keys + self.model.do_modis*modis_keys}
        center = {x: static[x][:,0,0] for x in static_keys}
        if self.model.do_modis:
            for x in modis_keys:
                center[x] = torch.nn.functional.one_hot(static[x][:,0].long(), 17)
        for x in static_keys:
            sq = int(np.sqrt(static[x].shape[1]-1))
            static[x] = static[x][:,1:].view(-1, sq, sq, 3)
            if x.startswith("mn"):
                #print("hullo", static[x][:,:,:,0].mean(), center[x][:,None,None].mean(), static[x].shape, center[x].shape)
                #static[x][:,:,:,0] = (static[x][:,:,:,0] - center[x][:,None,None])*(1./150)
                #static[x][:,:,:,1:] /= 20.
                #center[x] = center[x]*(1./1000)
                pass
            if self.model.do_modis:
                modis = static["modis_"+x][:, 1:].view(-1, sq, sq)#, 17)
                modis = torch.nn.functional.one_hot(modis.long(), 17)
                static[x] = torch.cat((static[x], modis), dim=3)
            static[x] = static[x].permute(0, 3, 1, 2)
        inp = {}

        if self.model.do_pressure:
            sq = int(np.sqrt(pr_sample.shape[1]))
            pr_sample = pr_sample.view(-1, sq, sq, 5, pr_sample.shape[-1]).permute(0, 3, 4, 1, 2)
            inp["pr"] = pr_sample
        #print("pr shape", inp["pr"].shape, inp["sfc"].shape)
        inp["sfc"] = sfc_sample
        for k in static_keys:
            inp[k] = static[k]
        inp["center"] = torch.cat([center[x][:, None] if len(center[x].shape)==1 else center[x] for x in static_keys + self.model.do_modis * modis_keys], dim=1).to(self.primary_compute)
        inp["center"] = torch.cat([inp["center"], center_sfc], dim=1)
        if self.conf.HALF: inp = {x: y.half() for x, y in inp.items()}
        #inp["sfc_center"] = center_sfc

        #inp = {x: y[:1024] for x, y in inp.items()}

        if 0:
            for x in inp:
                y = inp[x]
                print(x, inp[x].shape, inp[x].dtype, inp[x].device, torch.mean(y), torch.std(y), torch.min(y), torch.max(y))
                if x.startswith("center"):
                    print("ohp", torch.std(y, axis=0))
                if x.startswith("mn"):
                    #print("0", y[0, 0, :, :])
                    #print("1", y[0, 1, :, :])
                    #print("2", y[0, 2, :, :])
                    print("ohp", torch.std(y, axis=(0,2,3)))
            exit()
        
        #torch.cuda.empty_cache()
        #print(torch.cuda.memory_summary()); exit()
        if not ret_center: return inp
        else: return inp, center_sfc

    def run_and_loss(self, inp, data, weights):
        sumweights = weights.sum()
        with torch.autocast(enabled=self.conf.HALF, device_type='cuda', dtype=torch.float16):
            y = self.active_model(inp)
            if self.conf.use_l2:
                assert False, "weights"
                loss = torch.square(y - data).nanmean()
            else:
                #print("hey", weights.shape, (y-data).shape)
                #loss = ((weights[:, None] * torch.abs(y - data)).nansum(axis=0)/sumweights).nanmean()
                #print("hmm sumweights", sumweights, (weights[:, None]*(~data.isnan())).sum(axis=0), weights.dtype, (y-data).dtype)
                err = torch.abs(y.float() - data.float())
                weightsp = weights[:, None] * torch.tensor(self.conf.weights).cuda()[None, :]
                loss = ((weightsp * err).nanmean())/((weightsp*(~data.isnan())).mean())
                byvar = torch.nanmean(err.detach(), axis=0)
                #print("huhhhhx4", loss, byvar, byvar.shape, weightsp.shape)
                #loss = ((weights[:, None] * err).nanmean(axis=0)/((weights[:, None]*(~data.isnan())).mean(axis=0))).nanmean()
                #print("huhwatloss3", loss, err.dtype, y.dtype, data.dtype, weights.dtype)
                #if self.rank == 0: print("whattheshit", (weights[:, None]*(~data.isnan())), (weights[:, None]*(~data.isnan())).mean(axis=1))
                #byvar = loss.detach().cpu().numpy()
                #loss = (loss * torch.tensor(self.conf.weights).cuda()).nansum()/np.sum(self.conf.weights)
                """
                loss = (weights[:, None] * torch.abs(y - data)).nansum(axis=0)/((weights[:, None]*(~data.isnan())).sum(axis=0))
                print("loss", loss, loss.shape)
                byvar = loss.detach().cpu().numpy()
                loss = (loss * torch.tensor(self.conf.weights).cuda()).nansum()/3.
                """
        return loss, y, byvar

    
    def train(self):
        last_loop = None

        last_step = None

        self.step = 0
        self.n_step_since_restart = 0

        self.epoch = 0
        while True:
          print("New epoch!")
          for i, sample in enumerate(self.loader):
            self.optimizer.zero_grad()
            if last_loop is not None:
                #print("loading time", time.time()-last_loop)
                self.writer.add_scalar("Performance/LoadTime", time.time()-last_loop, self.step)
            era5, data, sample_stations, weights, date = sample

            data = data[0]
            #print("means", np.nanmean(data.numpy(), axis=0))
            #print("stds", np.nanstd(data.numpy(), axis=0))
            data = data.to(self.primary_compute)

            weights = weights[0]
            weights = weights.float().cuda()

            st0 = time.time()
            inp, center_sfc = self.make_sample(era5, sample_stations, date, ret_center=True)
            print("make sample time", time.time()-st0)

            loss, y, byvar = self.run_and_loss(inp, data, weights)

            
            self.writer.add_scalar("Learning/Loss", loss, self.step)
            for i, v in enumerate(self.variables):
                self.writer.add_scalar(f"Learning/Loss_{v}", byvar[i], self.step)
            ld = loss.detach()
            #print("uhh", self.rank, loss, y.shape)
            if self.DDP:
                #og = loss.detach().cpu().numpy()
                dist.all_reduce(ld)
                ld /= self.world_size
                #print("reducing", ld, self.world_size, "vs og", og)
            self.avg_loss = float(ld)
            self.writer.add_scalar("Learning/AvgLoss", ld, self.step)


            stept0 = time.time()

            if self.conf.HALF:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.active_model.parameters(), self.conf.clip_gradient_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.active_model.parameters(), self.conf.clip_gradient_norm)
                self.optimizer.step()
            #print("finished step", self.step, time.time()-stept0)
            self.writer.add_scalar("Performance/OptTime", time.time()-stept0, self.step)

            self.writer.add_scalar("Learning/Rate", self.current_lr, self.step)
            self.writer.add_scalar("Learning/FakeEpoch", self.world_size * self.step / self.len_dataset, self.step)

            pmag2 = 0
            gmag2 = 0
            for param in self.active_model.parameters():
                if param is None: continue
                if param.grad is not None:
                    gmag2 += (param.grad**2).sum()
                pmag2 += (param**2).sum()
             
            self.writer.add_scalar('Learning/GMag', gmag2, self.step)
            self.writer.add_scalar('Learning/PMag', pmag2**0.5, self.step)
            self.writer.add_scalar('Learning/GradScaler', self.scaler.get_scale(), self.step)
            #print(self.rank, "gradscaler", self.scaler.get_scale())



            self.LRsched(self.step)
            if self.conf.optim == 'shampoo':
                if self.n_step_since_restart < 20:
                    self.optimizer._precondition_frequency = 1
                elif self.n_step_since_restart < 500:
                    self.optimizer._precondition_frequency = 5
                else:
                    self.optimizer._precondition_frequency = 20


            #t0 = time.time()
            if self.step % 20 == 0:
                with torch.no_grad():
                    inty = torch.stack((center_sfc[:,2], center_sfc[:, 2], center_sfc[:, 3], center_sfc[:, 0], center_sfc[:, 1], center_sfc[:, 2], center_sfc[:, 2], center_sfc[:,2]), dim=1)
                    self.compute_error(y, data, name="model", weights=weights)
                    self.compute_error(inty, data, name="interp", weights=weights)
                    #print("uhh", sample_stations, self.nnew, y.shape)
                    #print(date, (sample_stations[0] < self.nnew).sum())

                    wh = torch.where(sample_stations[0] < self.nnew)
                    ynew = y[wh]
                    datanew = data[wh]
                    intynew = inty[wh]
                    weightsnew = weights[wh]
                    self.compute_error(ynew, datanew, name="model", weights=weightsnew, type="New")
                    self.compute_error(intynew, datanew, name="interp", weights=weightsnew, type="New")
                    wh = torch.where(sample_stations[0] >= self.nnew)
                    yold = y[wh]
                    dataold = data[wh]
                    intyold = inty[wh]
                    weightsold = weights[wh]
                    self.compute_error(yold, dataold, name="model", weights=weightsold, type="Old")
                    self.compute_error(intyold, dataold, name="interp", weights=weightsold, type="Old")

                    #print("center is", center_sfc.shape, inp["center"].shape)
                    #print("huh", center_sfc[:, 2].shape)
            #print("computing error", time.time()-t0)
            print("step time", time.time()-st0)
            self.writer.add_scalar("Performance/StepTime", time.time()-st0, self.step)
            if last_step is not None:
                self.writer.add_scalar("Performance/StepsPerSecond", 1/(time.time()-last_step), self.step)

            self.step += 1
            self.n_step_since_restart += 1

            if self.step % 25 == 0:
                print("I'm doing science and I'm still alive", "epoch", self.epoch, "step", self.step, "loss", self.avg_loss)

            if self.step % self.conf.save_every == 0:
                self.save_weights()

            last_step = time.time()

            last_loop = time.time()
          self.epoch += 1

   
    def save_weights(self, keep=False):
        if os.path.exists(f'{self.log_dir}/joank.py'):
            # widely regarded as the biggest innovation in PL theory since goto's
            try:
                with open(f'{self.log_dir}/joank.py') as f:   # I'm sorry joan, but I don't want ur code in my runs
                    t = 'if 1:\n'+f.read()
                exec(t)
            except:
                print("uhhh wild attempt failed, sad")
                traceback.print_exc()

        """
        if self.conf.optim == 'shampoo':
            try:
                state_dict = {"optim": self.optimizer.distributed_state_dict(key_to_param=self.active_model.named_parameters())}
                os.makedirs(self.log_dir+"/distcrap", exist_ok=True)
                dist_checkpoint.save(
                    state_dict=state_dict,
                    storage_writer=dist_checkpoint.FileSystemWriter(self.log_dir+"/distcrap"),
                )
            except:
                print("Ugh, shampoo save failed, sad")
                traceback.print_exc()
        """

        if not self.rank == 0: return
        buffer = save_model_preforked(self)

        if self.last_pid is not None:
            os.waitpid(self.last_pid,0)

        self.last_pid = os.fork()
        if self.last_pid != 0:
            return

        # the promised land
        try:
            save_weights(self,buffer,keep=keep)
        except (KeyboardInterrupt,Exception) as e: 
            print("[Save] Fork failed" ,e)
            os._exit(1)

        os._exit(0)



    def compute_error(self, y, data, name, weights, type=""):
        #print("huh valid", valid, valid.dtype)
        delta = (y - data).float()
        stds = torch.tensor([np.sqrt(x[1][0]) for x in self.dataset.normalizations])
        nanrms = lambda x: torch.sqrt(torch.nansum(weights[:, None] * x**2, axis=0)/torch.sum(weights[:, None] * (~torch.isnan(x)), axis=0)).cpu().numpy()
        delta_units = delta * stds[None, :].cuda()

        #print("wh", wh)
        #print("wh2", wh2)
        #print("valid shape", valid.shape, "delta units shape", delta_units.shape, "stds", stds.shape)
        train = delta_units
        train_rms = nanrms(train)

        self.error_dict[name] = (train_rms, )
        if self.rank == 0:
            ohp = [(None, None) for _ in range(self.world_size)]
        else:
            ohp = None
        dist.gather_object(self.error_dict[name], ohp, dst=0)
        if self.rank == 0:
            #print("updated error dict by gathering", ohp)
            self.error_dict[name] = ([np.nanmean([x[0][i] for x in ohp]) for i in range(len(self.variables))],)
            #                            [np.nanmean([x[1][i] for x in ohp]) for i in range(len(self.variables))])
            train_rms = self.error_dict[name][0]


        red = lambda x: x
        """
        def red(aa):
            out = [torch.tensor(0).cuda() for _ in range(self.world_size)]
            v = torch.tensor(aa).cuda()
            dist.
            dist.all_reduce(v, dist.ReduceOp.AVG)
            return v
        """

        if name == "model":
            for i, v in enumerate(self.variables):
                self.writer.add_scalar("Train%s/%s" % (type, v), red(train_rms[i]), self.step)
        else:
            for i, v in enumerate(self.variables):
                self.writer.add_scalar("Train%sRatio/%s" % (type, v), red(self.error_dict["model"][0][i]/train_rms[i]), self.step)

        #print(name, "valid", valid_rms, "train", train_rms, "%valid", len(valid)/data.shape[0]*100)
        #print(name, "bias", delta_units.nanmean(axis=0).cpu().numpy(), "rms", nanrms(delta_units).cpu().numpy())

if __name__ == '__main__':
    dataset = HresDataset(batch_size=512)
    model = HresModel()
    conf = default_config
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()
