if 1:
        # Same but Joan
        mesh = meshes.LatLonGrid(subsamp=2, levels=levels_small)
        model = ForecastStepSwin(mesh)
        #args.resume = 'ignored/runs/run_Sep3-John-12hr-retry_20230903-222501/model_epoch0_iter19200_loss0.068.pt'
        w = WeatherTrainer(args,mesh,model)
        w.conf.DH = 12
        w.conf.cpu_only = 1 
        
        w.setupData()
        d = enumerate(w.data_loader)
        i,b = next(d)

        # if only there was a bettery way to repeat code
        dummy1 = []
        dummy2 = []
        dummy3 = []
        dummy4 = []

        attns = []
        def hook(model,input,output):
            global attns
            attns.append(input)

        def hook1(model,input,output):
            global dummy1
            dummy1.append(input)

        def hook2(model,input,output):
            global dummy2
            dummy2.append(input)

        def hook3(model,input,output):
            global dummy3
            dummy3.append(input)

        def hook4(model,input,output):
            global dummy4
            dummy4.append(input)


        for i,m in enumerate(w.model.swin1.blocks):
            m.attn.attn_drop.register_forward_hook(hook)
            m.attn.dummy1.register_forward_hook(hook1)
            m.attn.dummy2.register_forward_hook(hook2)
            m.attn.dummy3.register_forward_hook(hook3)
            m.attn.dummy4.register_forward_hook(hook4)
        y,_ = w.forwards(b)
        w.computeErrors(b,y)
        import pdb; pdb.set_trace()
        #w.setupData()

