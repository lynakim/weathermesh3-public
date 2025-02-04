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
from neopointy.hres_utils import *

from neopointy.model import HresModel

from torch.utils.data.sampler import WeightedRandomSampler

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.utils.shampoo_utils import GraftingType

import torch.utils.checkpoint as checkpoint
import torch.distributed.checkpoint as dist_checkpoint

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch.utils.tensorboard import SummaryWriter

station_list = pickle.load(open("/huge/ignored/hres/station_list.pickle", "rb"))

def am_i_torchrun():
    return 'TORCHELASTIC_ERROR_FILE' in os.environ.keys()

def sillycolate(batch):
    return batch[0]

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
    def __init__(self, conf, batch_size, model, simple=False):
        y1 = list(range(1979,2022))
        y2 = list(range(2011, 2021))
        for y in y2:
            y1.remove(y)
        self.dataset1 = HresDataset(batch_size=batch_size//2, years=y1)
        self.dataset2 = HresDataset(batch_size=batch_size//2, years=y2)

        sampler1 = WeightedRandomSampler(np.array(self.dataset1.weights)**(1/2.), num_samples=len(self.dataset1.weights), replacement=True)
        self.loader1 = DataLoader(self.dataset1, batch_size=1, num_workers=2, sampler=sampler1, collate_fn=sillycolate)

        sampler2 = WeightedRandomSampler(np.array(self.dataset2.weights)**(1/2.), num_samples=len(self.dataset2.weights), replacement=True)
        self.loader2 = DataLoader(self.dataset2, batch_size=1, num_workers=2, sampler=sampler2, collate_fn=sillycolate)


        self.last_pid = None
        self.writer = NoOp()
        self.error_dict = {}
        self.B = 512
        self.variables =  ["tmpf", "dwpf", "mslp", '10u', '10v', 'skyl1', 'vsby', 'precip']
        self.batch_size = batch_size

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
        print("yoooo", self.gpus, self.local_rank)
        self.num_gpus = len(self.gpus)

        print("number of parameters", sum([x.numel() for x in self.model.parameters()])/1e6, "M")
        #self.primary_compute = torch.device("cuda")
        #self.model = self.model.to(self.primary_compute)
        #if HALF:
        #    self.model = self.model.half()
        #self.model = self.model.half()

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
        #print("before literally anything!!!", self.local_rank, self.gpus)
        #sys.stdout.flush()
        #time.sleep(10)
        c = self.conf
        self.rank = 0
        self.world_size = 1
        #self.primary_compute = torch.device(self.gpus[0]) 
        if self.DDP:
            dist.init_process_group('nccl')
            assert len(self.gpus) == 1, "DDP only works with one GPU per process"
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.primary_compute = torch.device(self.gpus[0])
            torch.cuda.set_device(self.primary_compute)
            self.num_gpus = self.world_size
            print("Starting DDP with rank",self.rank, "world size", self.world_size, "compute", self.primary_compute)
        #elif self.conf.cpu_only:
        #    self.primary_compute = torch.device('cpu')

        torch.manual_seed(time.time()*self.rank + self.rank + 42./(self.rank+1))

        """
        print("before making datasets!!!", self.local_rank, self.rank)
        print("before making datasets!!!", self.local_rank, self.rank)
        sys.stdout.flush()
        time.sleep(10)
        """

        self.len_dataset = len(self.dataset1)+len(self.dataset2)
        
        if '_' in self.conf.name: print("WARNING: underscore in name messes with things, replacing with dash")
        self.conf.name = self.conf.name.replace('_','-')

        self.log_dir = (self.conf.nope*"/tmp/")+f"{RUNS_PATH}/runs_hres/run_{self.conf.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"  
        if self.DDP:
            brdcast = [self.log_dir if self.rank == 0 else None]
            dist.broadcast_object_list(brdcast, src=0)
            self.log_dir = brdcast[0]

        """
        print("before moving model!!!", self.local_rank, self.rank)
        print("before moving model!!!", self.local_rank, self.rank)
        sys.stdout.flush()
        time.sleep(10)
        """

        self.model = self.model.to(self.primary_compute)

        if self.DDP:
            self.active_model = DistributedDataParallel(self.model,device_ids=[self.primary_compute])#,find_unused_parameters=True)
        else:
            self.active_model = self.model

        if self.rank != 0: return

        self.writer = SummaryWriter(log_dir=self.log_dir)


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

 
    def run_and_loss(self, inp, squares):#normcoords, data, weights):
        #sumweights = weights.sum()

        bilinear_vars = inp["era5sfc"][:,:,:,[2, 7, 3, 0, 1, 2, 2, 2]]
        bilinear_vars[..., 5] = 0
        bilinear_vars[..., 6] = 0
        bilinear_vars[..., 7] = 0
        bilinear = torch.nn.functional.interpolate(bilinear_vars.permute(0,3,1,2), scale_factor=30, align_corners=True, mode='bilinear')
        del bilinear_vars
        #import pdb; pdb.set_trace()
        with torch.autocast(enabled=self.conf.HALF, device_type='cuda', dtype=torch.float16):
            y = self.active_model(inp)
            total_err = []
            errs = []
            sumweights = 0
            sumN = 0
            bilinear_err = []
            model_err = []

            #self.variables = variables or ["tmpf", "dwpf", "mslp", '10u', '10v', 'skyl1', 'vsby', 'precip']
            fac = torch.tensor([0.25, 0.25, 0.15, 0.37, 0.37, 1, 1, 1]).float().to(self.primary_compute)
            t0 = time.time()
            for i, (data, normcoords, weights) in enumerate(squares):
                #dt = time.time()-t0
                #torch.cuda.empty_cache()
                #mem = torch.cuda.mem_get_info()
                #print("starting square", i, "mem", mem[0]/(1024**3), mem[1]/(1024**3), "took", dt)

                weights = weights.to(self.primary_compute)
                weightsp = weights[:, None] * torch.tensor(self.conf.weights).to(self.primary_compute)[None, :]
                weightsp = weightsp[None]
                sw = float(weightsp.float().nansum().item())
                sumweights += sw
                #sumN += normcoords.shape[0]
                assert normcoords.shape[0] == data.shape[0]
                assert normcoords.shape[1] == 2
                ######normcoords = torch.fliplr(normcoords) # god i hope not
                normcoords = normcoords[None][None].to(self.primary_compute)
                out = y[i:i+1].permute(0,3,1,2)
                interp = torch.nn.functional.grid_sample(out, normcoords, mode='bilinear', align_corners=True, padding_mode="border")
                interp = interp[:, :, 0, :].permute(0, 2, 1)

                assert out.shape == bilinear[i:i+1].shape
                bilinear_interp = torch.nn.functional.grid_sample(bilinear[i:i+1], normcoords, mode='bilinear', align_corners=True, padding_mode="border")
                bilinear_interp = bilinear_interp[:, :, 0, :].permute(0, 2, 1)

                target = (data - bilinear_interp[0])/fac[None, :]


                predicted = bilinear_interp + interp * fac[None, :]
                # pred = bilinear_interp + fac * interp
                # err = (bilienar_interp + fac * interp - ref)
                # err = 0 --> (ref - bilinear_interp)/fac = interp

                #print("bilienar_interp", bilinear_interp.shape, "out", out.shape)
                bilinear_err.append(bilinear_interp[0] - data)
                ###model_err.append((interp[0] - data[0]).detach())
                model_err.append((predicted[0] - data).detach())
                #print("got", interp.shape, data.shape)
                ###err = torch.abs(interp.float() - data.float())[0]
                #err = torch.abs(target)
                #print("shapes err", err.shape, "weights", weights.shape)
                #errs.append(err.detach())
                err2 = torch.abs(interp[0] - target)
                total_err.append((torch.nanmean(err2*weightsp), sw))

                #dt = time.time()-t0
                #torch.cuda.empty_cache()
                #mem = torch.cuda.mem_get_info()
                #print("finished square", i, "mem", mem[0]/(1024**3), mem[1]/(1024**3), "took", dt)

            del predicted, target, interp, bilinear_interp
            rankweights = float(sumweights)
            rankN = float(sumN)
            sumweights = torch.tensor(rankweights).to(self.primary_compute)
            #sumN = torch.tensor(rankN).cuda()
            dist.all_reduce(sumweights, op=dist.ReduceOp.SUM)
            #dist.all_reduce(sumN, op=dist.ReduceOp.SUM)

            #print("rank", self.rank, "sumweights", rankweights, "->", sumweights)#, "sumN", rankN, "->", sumN)
            bilinear_err = torch.cat(bilinear_err, axis=0)
            model_err = torch.cat(model_err, axis=0)

            loss = 0
            for x, y in total_err:
                if not torch.isnan(x):
                    loss += x * (self.world_size * y/sumweights)
            #loss = torch.sum(torch.tensor([x*y/sumweights for x, y in total_err])).cuda()
            #print(loss)
            #print(errs[0].shape, errs[1].shape)
            #errs = torch.cat(errs, axis=0)
            #byvar = torch.nanmean(errs, axis=0)

            byvar = torch.mean(model_err, axis=0)


            if torch.isnan(loss):
                print("rank", self.rank, "loss is nan", loss, "byvar", byvar)
                print("weights", rankweights, "sumweights", sumweights, "ww", [(x,y) for x, y in total_err])
                print("also", [x[0].shape for x in squares])
            #print("byvar", byvar.shape)

            dt = time.time()-t0
            #torch.cuda.empty_cache()
            #mem = torch.cuda.mem_get_info()
            #print("finished function call", "mem", mem[0]/(1024**3), mem[1]/(1024**3), "took", dt)
            return loss, model_err, bilinear_err, byvar
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

        profile = True
        profile = False
        if profile:
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=2, warmup=2, active=2, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            )
            prof.start()

        iter1 = iter(self.loader1)
        iter2 = iter(self.loader2)
        """
        print("sleeeeeeeping before we start!!!", self.local_rank)
        print("sleeeeeeeping before we start!!!", self.local_rank)
        import sys
        sys.stdout.flush()
        print("sleeeeeeeping before we start!!!", self.local_rank)
        sys.stdout.flush()
        time.sleep(10)
        print("done sleeping we start!!!", self.local_rank)
        """

        i = 0
        while True:
            #print("New epoch!")
            #for i, sample in enumerate(self.loader):
            xt0 = time.time()
            try:
                sample1 = next(iter1)
            except:
                iter1 = iter(self.loader1)
                sample1 = next(iter1)
            try:
                sample2 = next(iter2)
            except:
                iter2 = iter(self.loader2)
                sample2 = next(iter2)
            
            yt0 = time.time()
            squares = sample1[0] + sample2[0]
            dic = {k: torch.cat([sample1[1][k], sample2[1][k]], axis=0) for k in sample1[1]}
            del sample1
            del sample2
            #print("merging samples took", time.time()-xt0, time.time()-yt0)

            """
            if profile:
                prof.step()
                if i == 1:
                    if self.rank == 0: torch.cuda.memory._record_memory_history(max_entries=100000)
                if i == 5:
                    if self.rank == 0: torch.cuda.memory._dump_snapshot("/fast/memory_snapshot2.pickle")
                    prof.stop()
                    exit()
            """
            self.optimizer.zero_grad()
            if last_loop is not None:
                #print("loading time", time.time()-last_loop)
                self.writer.add_scalar("Performance/LoadTime", time.time()-last_loop, self.step)
            #import pdb; pdb.set_trace()

            #squares, dic = sample
            dic = {k: v.to(self.primary_compute) for k, v in dic.items()}
            #print("hiya", self.rank, self.local_rank, dic["era5sfc"].device)
            #for k in dic:
            #    print("dic", k, dic[k].shape, dic[k].dtype, torch.mean(dic[k].float(), axis=(0,1,2)), torch.std(dic[k].float(), axis=(0,1,2)))
            #import pdb; pdb.set_trace()

            squares = [[y.to(self.primary_compute) for y in x] for x in squares]
            #for i, sq in enumerate(squares):
            #    for y in sq:
            #        print("sq", y.shape, y.dtype)
            st0 = time.time()

            loss, model_err, bilinear_err, byvar = self.run_and_loss(dic, squares)
            del dic
            del squares
            
            self.writer.add_scalar("Learning/Loss", loss, self.step)
            #byvar = byvar.to(self.primary_compute)
            #dist.all_reduce(byvar, op=dist.ReduceOp.AVG)
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


            #torch.cuda.empty_cache()
            #mem = torch.cuda.mem_get_info()
            #print("prebackward", mem[0]/(1024**3), mem[1]/(1024**3))

            stept0 = time.time()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.active_model.parameters(), self.conf.clip_gradient_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            print("finished step", self.step, time.time()-stept0)
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

            print("took", time.time()-st0, torch.cuda.max_memory_reserved()/(1024**3), "GiB")

            #t0 = time.time()
            if self.step % 10 == 0:
                with torch.no_grad():
                    #inty = torch.stack((center_sfc[:,2], center_sfc[:, 2], center_sfc[:, 3], center_sfc[:, 0], center_sfc[:, 1], center_sfc[:, 2], center_sfc[:, 2], center_sfc[:,2]), dim=1)
                    self.compute_error(model_err, name="model")#, weights=weights)
                    self.compute_error(bilinear_err, name="interp")#, weights=weights)
                    #print("uhh", sample_stations, self.nnew, y.shape)
                    #print(date, (sample_stations[0] < self.nnew).sum())

                    #print("center is", center_sfc.shape, inp["center"].shape)
                    #print("huh", center_sfc[:, 2].shape)
            #print("computing error", time.time()-t0)
            #print("step time", time.time()-st0)
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
            i += 1

   
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



    def compute_error(self, delta, name, type=""):
        #print("huh valid", valid, valid.dtype)
        delta = delta.float()
        stds = torch.tensor([np.sqrt(x[1][0]) for x in self.dataset1.normalizations])
        nanrms = lambda x: torch.sqrt(torch.nansum(weights[:, None] * x**2, axis=0)/torch.sum(weights[:, None] * (~torch.isnan(x)), axis=0)).cpu().numpy()
        nanrms = lambda x: torch.sqrt(torch.nanmean(x**2, axis=0)).cpu().numpy()
        delta_units = delta * stds[None, :].to(self.primary_compute)

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
                self.writer.add_scalar("Bilinear%s/%s" % (type, v), red(train_rms[i]), self.step)

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
