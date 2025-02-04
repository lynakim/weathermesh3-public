from train import *

if __name__ == '__main__':    
    args = parse_args()
    if args.dimprint:
        dimprint = print
    else:
        dimprint = lambda *a, **b: None

    torch.multiprocessing.set_start_method('fork')

    if args.activity == '1':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        M = 896
        model = ForecastStepSwin(mesh, transformer_input=M, num_heads=16, encoder_hidden=2*M)
        w = WeatherTrainer(args,mesh,model)
        w.conf.DH = 3
        w.conf.optim = torch.optim.AdamW
        #w.conf.adam.lr *= 2
        w.conf.adam.betas = (0.9, 0.95)
        w.conf.adam.weight_decay = 0.1
        w.conf.lr_sched.lr = 5e-4
        w.conf.lr_sched.warmup_end_step = 500
        w.conf.HALF = 1
        w.conf.save_every = 1000
        #w.conf.log_every = 2
        #w.conf.GRAD_ACCUM = 8 # XXX experiment
        #w.conf.adam.lr = 1e-4 # XXX experiment
        
        w.run()

    if args.activity == '13':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin3D(mesh)
        w = WeatherTrainer(args,mesh,model)
        w.conf.DH = 12
        w.conf.lr_sched.lr = 5e-5
        #w.conf.GRAD_ACCUM = 8 # XXX experiment
        #w.conf.adam.lr = 1e-4 # XXX experiment
        
        w.run()

    
    if args.activity == 'linear':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepLinear(mesh)
        w = WeatherTrainer(args,mesh,model)
        w.conf.dates = [D(2015, 1, 1)] 
        w.conf.dates = get_dates((D(2015, 1, 1),D(2015, 1,14))) 
        w.conf.only_at_z = [12]
        w.conf.save_every = 100
        w.conf.log_every = 20
        w.run()
    
    if args.activity == '3':
        # Messing around at analying a model
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin(mesh) 
        #model = ForecastStepLinear(mesh) 
        args.resume = 'ignored/runs/run_Sep7-Project1_20230907-010534/model_epoch3_iter23794_loss0.086.pt'
        w = WeatherTrainer(args,mesh,model)
        w.conf.DH = 3
        w.conf.cpu_only = 1 
        
        w.setup_data()
        d = enumerate(w.data_loader)
        i,b = next(d)
        import copy
        b_copy = copy.deepcopy(b)
        wx = w.unnorm(b)[0,1]

        attns = []
        def hook(model,input,output):
            global attns
            attn = input[0].cpu().detach().numpy().astype(np.float16)
            attns.append(attn)
        for i,m in enumerate(w.model.swin1.blocks):
            m.attn.attn_drop.register_forward_hook(hook)
        y,_ = w.forwards(b)
        w.computeErrors(b,y)
        
        with open('ignored/vis/attns.pickle','wb') as f:
            pickle.dump({'attns':attns,'mesh':mesh,'wx':wx},f)
        print(os.stat('ignored/vis/attns.pickle').st_size/1e9)

    if args.activity == '5':
        import anl
        import pickle
        print('hi'); t0 = time.time()
        with open('ignored/vis/attns.pickle','rb') as f:
            d = pickle.load(f)
        attns=d['attns']; mesh=d['mesh']; #wx=d['wx']
        w = WeatherTrainer(args,mesh,None)
        w.setup_data()
        b,date = w.weather_data[0]
        bx = b.numpy()
        print(b.shape,bx.shape,date)
        exit()
        import matplotlib.pyplot as plt
        wx = w.unnorm(b)[0,1]
        print('loaded')
        print('setup took:',time.time()-t0)
        import john2
        anl.anl_attn(attns,mesh,wx)
 

    if args.activity == 'spike':
        import anl
        import pickle

        tst = '2005-11-07 21:00:00Z'
        args.resume = 'ignored/runs/run_Sep11-Spikes_20230911-223512/model_epoch0_iter642_loss13.231.pt'
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin(mesh) 
        w = WeatherTrainer(args,mesh,model)
        w.conf.cpu_only = 0 
        w.setup_data()
        with torch.no_grad():
            date1 = datetime.strptime(tst,'%Y-%m-%d %H:%M:%S%z').astimezone(timezone.utc) + timedelta(hours=24)
            date2 = datetime.strptime(tst,'%Y-%m-%d %H:%M:%S%z').astimezone(timezone.utc) + timedelta(hours=72)
            b1,_ = w.weather_data[date1]
            b2,_ = w.weather_data[date2]
            b1[1] = b2[0]
            b = default_collate((b1,))
            y,_ = w.forwards(b)
            w.computeErrors(b,y)

    if args.activity == 'clean':
        
        run = 'ignored/runs/run_Sep11-Spikes_20230911-223512'
        clean_saves(run)

    if args.activity == '4':
        exec(open('joaaan.py').read())


    if args.activity == 'ddelta':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin(mesh) 
        w = WeatherTrainer(args,mesh,model)
        w.setup_data()
        with torch.no_grad():
            lst = list(range(len(w.weather_data)))
            #lst = [7430, 7431, 8142, 8150, 9054, 9062, 9902, 9910, 15838, 15845, 17726, 17734, 18678, 18686, 23262, 23270, 27285, 27286, 29038, 29046, 29406, 29414, 31230, 31238, 35374, 35382, 36542, 36550]
            for i in lst:
                print(i,len(w.weather_data))
                b,_ = w.weather_data[i]
                b = b.unsqueeze(0)
                #print(b.shape)
                r = w.computeErrors(b,b[:,2]-b[:,1],unnorm=False,do_print=0)
                er = r['129_z'][1]
                with open('ignored/er.txt','a') as f:
                    f.write(f'{i},{er}\n')

    if args.activity == 'ddelta2':
        w = get_trainer(args)
        w.conf.cpu_only = 1
        w.primary_compute = torch.device("cpu")
        with open('ignored/spikes1997.txt','r') as f:
            lines = f.readlines()
        lines = [l.strip().split(' ') for l in lines[1:]]
        tots = []
        for l in lines:
            t = float(l[1])
            tots.append(t)
            if t>200:
                i = int(l[0])
                b,d = w.weather_data[i]
                dt = get_date(d)
                b = b.unsqueeze(0)
                r = w.computeErrors(b,b[:,2]-b[:,1],unnorm=False,do_print=0)
                er = r['129_z'][1]
                print(i,dt,l[1],er)
        print(np.max(tots), np.percentile(tots, 90), np.mean(tots))

    if args.activity == 'ddelta2':
        w = get_trainer(args)
        with open('ignored/er.txt','r') as f:
            lines = f.readlines()
        lines = [l.strip().split(',') for l in lines]
        for l in lines:
            if float(l[1])>200:
                i = int(l[0])
                b,d = w.weather_data[i]
                dt = get_date(d)
                b = b.unsqueeze(0)
                r = w.computeErrors(b,b[:,2]-b[:,1],unnorm=False,do_print=0)
                er = r['129_z'][1]
                print(i,dt,l[1],er)



    if args.activity == 'specdelta':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin(mesh) 
        w = WeatherTrainer(args,mesh,model)
        w.conf.cpu_only = 1
        w.setup_data()
        with torch.no_grad():
            idxs = [9062]
            idxs = list(range(len(w.weather_data)))
            for i in idxs:#range(len(w.weather_data)):
                #print(i,len(w.weather_data))
                b,_ = w.weather_data[i]
                b = b.unsqueeze(0)
                #print(b.shape)
                r, err = w.computeErrors(b,b[:,2]-b[:,1],unnorm=False,do_print=0, full=True)
                aa = r["129_z"][1]
                if aa > 160:
                    print(i, aa)
                continue
                e = err[0]
                e.shape = (360, 720, 6, 14)
                #print(e[:,:,0])
                rms = np.sqrt(np.mean(np.square(e[:,:,0]), axis=(0,1)))
                #print("bylev", rms)

                er = r['129_z'][1]
                print("yo er", i, er)

    if args.activity == 'specdelta2':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin(mesh) 
        w = WeatherTrainer(args,mesh,model)
        w.conf.cpu_only = 1
        w.setup_data()

        def proc(i):
            print("eyo", i)
            try:
                b,_ = w.weather_data[i]
                b = b.unsqueeze(0)
                print(b.shape, _)
            except:
                import traceback
                traceback.print_exc()
        idxs = list(range(len(w.weather_data)))
        pool = multiprocessing.Pool(8)
        pool.map(proc, idxs)
        exit()
            
        with torch.no_grad():
            idxs = [9062]
            idxs = list(range(len(w.weather_data)))
            for i in idxs:#range(len(w.weather_data)):
                #print(i,len(w.weather_data))
                b,_ = w.weather_data[i]
                b = b.unsqueeze(0)
                print(b.shape, _)
                continue
                #print(b.shape)
                r, err = w.computeErrors(b,b[:,2]-b[:,1],unnorm=False,do_print=0, full=True)
                print(i, r["129_z"][1])
                continue
                e = err[0]
                e.shape = (360, 720, 6, 14)
                #print(e[:,:,0])
                rms = np.sqrt(np.mean(np.square(e[:,:,0]), axis=(0,1)))
                #print("bylev", rms)

                er = r['129_z'][1]
                print("yo er", i, er)

    if args.activity == 'log':
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin(mesh)
        w = WeatherTrainer(args,mesh,model)
        w.setup_data()
        w.setupLogging()
        w.setupTraining()
        for attr, value in w.__dict__.items():
            if isinstance(value, (bool, int, float, str, SimpleNamespace)):
                print(attr)