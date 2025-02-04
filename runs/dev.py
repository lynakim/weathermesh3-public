from launch import * 

if am_i_torchrun():
    from train import *

from train import * # <--- comment this back in if not using DDP

@launch(ddp=0)
def memtesting2():
    memmon = MemoryMonitor()
    def pmem(tt):
        nonlocal memmon
        print(f"Sum {sum([size_mb(t) for t in tt]) : 0.1f} MiB")
        print(f"RSS {memmon.get_main_rss() / 2**20 : 0.1f} MiB") 
    ts = []
    for i in range(5):
        t = torch.zeros(1024+1,1024,1024,dtype=torch.float16,pin_memory=True)
        #print(f"addr: 0x{t.data_ptr():x}")
        ts.append(t)
        pmem(ts)




@launch(ddp=0)
def memtesting():
    import matplotlib.pyplot as plt
    memmon = MemoryMonitor()
    def pmem(tt):
        nonlocal memmon
        print(f"Sum {sum([size_mb(t) for t in tt]) : 0.1f} MiB")
        print(f"RSS {memmon.get_main_rss() / 2**20 : 0.1f} MiB")    

    rss = []
    sm = [] 
    def sv(tt):
        rss.append(memmon.get_main_rss() / 2**20)
        sm.append(sum([size_mb(t) for t in tt]))

    tgpus = []
    tcpus = []
    for i in range(20):
        t = torch.zeros(129,1024,1024,dtype=torch.float16)
        size = size_mb(t)
        tgpu = t.to('cuda')
        del t 
        tgpus.append(tgpu)
        #pmem(tgpus)
   
    print("BACK TO CPU")
    for i in range(20):
        tcpu = tgpus[i].to('cpu',non_blocking=False)
        tcpu = tcpu.pin_memory()
        print(f"pinned: {tcpu.is_pinned()}")
        tcpus.append(tcpu)
        pmem(tcpus)
        sv(tcpus)
    #return
    plt.grid()
    plt.title(f"Size: {size:0.2f} MiB")
    plt.plot(rss,label='RSS')
    plt.plot(sm,label='Sum')
    plt.legend()
    plt.savefig(f'john/memtest_blocking_pin{size:0.2f}.png')


@launch(ddp=0,start_method='spawn')
def Jul21_datalog():
    config.nope = True
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 1, 1), D(2022, 7, 1))])
    vdates = get_dates((D(2020, 1, 1),D(2020, 12, 28)))
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28',extra_sfc_vars=extra, output_only_vars=extra, is_output_only=True,levels=levels_medium)


    timesteps = [24]
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh], outputs=[mesh],timesteps=timesteps,worker_complain = False,requested_dates = tdates,max_ram_manual = int(6e9)))
                                           #only_at_z=[0,6,12,18]
    model = ForecastStepSwin3D(
        ForecastStepConfig(
            data.config.inputs, outputs=data.config.outputs, patch_size=(4,8,8), window_size=(2,3,6), 
            hidden_dim=1024, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=8, lat_compress=False, 
            timesteps=timesteps, dims_per_head=32, processor_dt=3, use_matepoint=True, output_deltas=True, 
            decoder_reinput_initial=True, decoder_reinput_size=96, neorad=True))
    model.output_deltas = True
    model.do_sub = False
    w = WeatherTrainer(conf=config,model=model,data=data)
    w.preserved_conf.validate_every = 300_000_000
    w.preserved_conf.ignore_train_safegaurd = True
    #w.preserved_conf.save_every = 5
    w.preserved_conf.DH = 24
    w.preserved_conf.validate_N = 8
    w.preserved_conf.log_every = 50
    #w.preserved_conf.save_every = 1
    #w.preserved_conf.use_shampoo = True
    w.preserved_conf.optim = 'adam'
    #w.preserved_conf.reset_optimizer = True
    w.preserved_conf.lr_sched = SimpleNamespace()
    w.preserved_conf.lr_sched.cosine_period = 45_000
    w.preserved_conf.lr_sched.cosine_bottom = 5e-8
    w.preserved_conf.lr_sched.warmup_end_step = 1_000
    w.preserved_conf.lr_sched.lr = 0.33e-3
    w.preserved_conf.adam = SimpleNamespace()
    w.preserved_conf.adam.betas = (0.9, 0.99)
    w.preserved_conf.adam.weight_decay = 0.001
    w.preserved_conf.lr_sched.step_offset = 0
    #w.preserved_conf.dates = get_dates((D(1971, 1, 1),D(2017, 12,30)))
    w.preserved_conf.dates = tdates
    w.preserved_conf.val_dates = vdates
    w.run()

@launch(ddp=False,nodes={'martins':0},port=29500)
def exploration():

    mesh = meshes.LatLonGrid(WEATHERBENCH=1, CLOUD=0,source='era5-28')
    data = NeoWeatherDataset(NeoDataConfig(inputs=[mesh],
                                           outputs = [mesh],
                                           timesteps=[24],
                                           requested_dates = get_dates((D(2021, 3, 20),D(2022, 7, 30))),
                                           ))
    model = ForecastStepSwin3D(ForecastStepConfig(inputs=[mesh], patch_size=(4,8,8), hidden_dim=768, enc_swin_depth=4, dec_swin_depth=4, proc_swin_depth=12, 
                                                  lat_compress=False, timesteps=[24], dims_per_head=32, processor_dt=12,use_matepoint=False)).half().cuda()
    data.check_for_dates()
    sample = default_collate([data[0]])
    x = sample[0]
    x = [xx.cuda() for xx in x]
    model(x,dt=24)


@launch(ddp=True,nodes={'miramar':6},port=29500,log=False)
def nccl_inq():
    import torch.distributed as dist
    dist.init_process_group('nccl',timeout=timedelta(seconds=3))
    #get rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("Hi from rank",rank,"of",world_size)
    torch.cuda.set_device(rank)

    data = torch.randn(1000,1000).cuda()
    if rank == 1: time.sleep(1)

    if rank == 0: 
        time.sleep(0.1)
        gathered = [torch.zeros_like(data) for _ in range(world_size)]
        dist.gather(data,gathered)
    else:
        dist.gather(data,dst=0)

    print(f"{rank} DONE!!!")
    sys.exit(0)




if __name__ == '__main__':
    run(locals().values())