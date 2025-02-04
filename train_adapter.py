from utils import *
from model_gfs_adapter import *
from train import *
from torch.utils.data import DataLoader    


def mod_config_for_adapter(config):
    self = SimpleNamespace()
    self.lr_sched = SimpleNamespace()
    self.lr_sched.lr = 2e-4
    self.lr_sched.warmup_end_step = 50
    self.lr_sched.step_offset = 0
    self.lr_sched.cosine_period = 20_000
    self.lr_sched.cosine_en = 1
    self.lr_sched.div_factor = 2
    self.lr_sched.cosine_bottom = None
    self.adam = SimpleNamespace()
    self.adam.betas= (0.9, 0.995)
    self.adam.weight_decay= 1e-8
    update_namespace(config,self)
    return config

class AdapterTrainer(WeatherTrainerBase):
    def __init__(self,data,model,adapter,config=None):
        if config == None:
            config = WeatherTrainerConfig()
        config = mod_config_for_adapter(config)
        
        self.conf = config ; c = config
        self.data = data
        self.model = model
        self.adapter = adapter
        self.state = SimpleNamespace()
        self.state.n_steps = 0
        self.state.n_iters = 0

        #jank


    def prep(self):
        c = self.conf
        self.primary_copmute = torch.device(f'cuda:{self.conf.gpus[0]}')

        self.model.eval()
        self.active_model = self.model.to(self.primary_copmute)
        self.active_adapter = self.adapter.to(self.primary_copmute)

        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(self.conf.HALF))
        self.scaler.set_growth_interval(1000)
        self.criterion = nn.MSELoss()

        self.data.check_for_dates()
        #set seed
        torch.manual_seed(0)
        
        self.dataloader = DataLoader(self.data, shuffle=True,batch_size=self.conf.batch_size, num_workers=4,prefetch_factor=2)

        #self.dataloader = NeoLoader(self.data)

        self.optimizer = torch.optim.AdamW(self.active_adapter.parameters(), lr=5e-4, **vars(self.conf.adam))


        self.log_dir = (c.nope*"/tmp/")+f"{RUNS_PATH}/runs_adapt{c.prefix}/run_{c.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(self.log_dir,exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        self.LRsched(self.state.n_iters)


    
    def run(self):
        self.prep()
        while True:
            for i,data in enumerate(self.dataloader):
            #while True:
                #data = self.dataloader.get_sample_now() 

                self.state.n_iters += 1
                nix = int(data[0][1].item())
                x_gfs = data[0][0].to(self.primary_copmute)
                x_era5 = data[1][0]
                x_era5 = self.data.extend_and_clamp_input(x_era5,get_date(nix)).to(self.primary_copmute)

                yt_era5 = data[2][0].to(self.primary_copmute)
                dt = int(data[2][1].item()) // 3600
                assert dt == 24

                with torch.autocast(enabled=bool(self.conf.HALF), device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        target = self.model.conv_forward(x_era5).detach()
                    y = self.adapter.forward(x_gfs)
                    loss = self.criterion(y,target)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.state.n_steps += 1
                    self.LRsched(self.state.n_steps)
                    self.writer.add_scalar('learning/loss',loss.item(),self.state.n_steps)
                    self.writer.add_scalar('learning/epochs',self.state.n_iters/len(self.dataloader),self.state.n_steps)
                    self.writer.add_scalar('learning/lr',self.optimizer.param_groups[0]['lr'],self.state.n_steps)

                if self.state.n_iters % 3 == 0:
                    with torch.no_grad():
                        with torch.autocast(enabled=bool(self.conf.HALF), device_type='cuda', dtype=torch.float16):
                            #yy = self.data.extend_and_clamp_input(y.to('cpu'),get_date(nix)).to(self.primary_copmute)
                            yp = self.model.forward(y,skip_conv=True,dt=dt)
                            x,yp = unnorm_output(x_era5,yp,self.model,dt)
                            yt = unnorm(yt_era5,self.model.config.inputs[0])
                        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float32):
                            rms = compute_errors(yp,yt,self.model.config.inputs[0])
                        print(f'{self.state.n_iters}, conv mse: {loss.item():.4f}, z500 rmse: {rms["129_z_500"]:.2f}')

                        self.writer.add_scalar('error/z500',rms["129_z_500"],self.state.n_steps)

                      










    