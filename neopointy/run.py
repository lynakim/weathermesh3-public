import sys
sys.path.append("../runs")
sys.path.append("..")
from launch import *
print("noimports!!")

@launch(nodes={"stinson": 3}, port=29500, start_method="spawn", log=True,clear_cache=False)
def Nov15_small():
    dataset = HresDataset(batch_size=22) # 650
    # factor, L, n_trans, wsize
    #model = HresModel(steps=[(30, 64, 4, 5)])
    model = HresModel(steps=[(6, 128, 6, 7), (5, 128, 6, 7)])
    #model = HresModel(steps=[(2, 128, 4, 9), (3, 128, 4, 9), (5, 128, 4, 9)])
    #model = HresModel(steps=[(6, 256, 4, 9) (5, 256, 4, 9)])
    #res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    #check = torch.load(res,map_location='cpu')
    #model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"stinson": 1}, port=29501, start_method="spawn", log=True,clear_cache=False, kill_nvidia=False)
def Nov14_neotest2_sillyflipped():
    dataset = HresDataset(batch_size=8) # 650
    # factor, L, n_trans, wsize
    #model = HresModel(steps=[(30, 64, 4, 5)])
    #model = HresModel(steps=[(2, 128, 4, 5), (3, 256, 4, 7), (5, 384, 6, 9)])
    model = HresModel(steps=[(2, 256, 4, 9), (3, 256, 4, 9), (5, 256, 4, 9)])
    #res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    #check = torch.load(res,map_location='cpu')
    #model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"stinson": 3}, port=29500, start_method="spawn", log=True,clear_cache=False, kill_nvidia=False)
def Nov14_neotest2_deltas():
    dataset = HresDataset(batch_size=11) # 650
    # factor, L, n_trans, wsize
    #model = HresModel(steps=[(30, 64, 4, 5)])
    #model = HresModel(steps=[(2, 128, 4, 5), (3, 256, 4, 7), (5, 384, 6, 9)])
    model = HresModel(steps=[(2, 256, 4, 9), (3, 256, 4, 9), (5, 256, 4, 9)])
    #res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    #check = torch.load(res,map_location='cpu')
    #model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"muir": 6}, port=29502, start_method="fork", log=True,clear_cache=False)
def Nov14_neotest2_multideltas_10xlr():

    import os, torch
    print("yooooo", os.environ.get('LOCAL_RANK'))
    #os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    from train import HresModel, Trainer
    #dataset = HresDataset(batch_size=40) # 650
    # factor, L, n_trans, wsize
    #model = HresModel(steps=[(30, 64, 4, 5)])
    L = 256
    L = 320
    model = HresModel(steps=[(2, L, 4, 9), (3, L, 4, 9), (5, L, 4, 9)], chunk_size=6)
    #res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    #check = torch.load(res,map_location='cpu')
    #model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    conf.optim = 'shampoo'
    conf.lr_sched.lr = 0.5e-3
    wt = Trainer(conf=conf, batch_size=36, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"muir": 6}, port=29503, start_method="fork", log=True,clear_cache=False)
def Nov14_neotest2_multideltas_dim32():

    import os, torch
    print("yooooo", os.environ.get('LOCAL_RANK'))
    #os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    from train import HresModel, Trainer
    #dataset = HresDataset(batch_size=40) # 650
    # factor, L, n_trans, wsize
    #model = HresModel(steps=[(30, 64, 4, 5)])
    L = 256
    L = 320
    model = HresModel(steps=[(2, L, 4, 9), (3, L, 4, 9), (5, L, 4, 9)], chunk_size=6, dims_per_head=32)
    #res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    #check = torch.load(res,map_location='cpu')
    #model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    conf.optim = 'shampoo'
    conf.lr_sched.lr = 1e-4
    wt = Trainer(conf=conf, batch_size=36, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"muir": 1}, port=29503, start_method="fork", log=True,clear_cache=False)
def Nov19_neotest2_pr():

    import os, torch
    print("yooooo", os.environ.get('LOCAL_RANK'))
    #os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    from train import HresModel, Trainer
    #dataset = HresDataset(batch_size=40) # 650
    # factor, L, n_trans, wsize
    #model = HresModel(steps=[(30, 64, 4, 5)])
    L = 256
    L = 320
    model = HresModel(steps=[(2, L, 4, 9), (3, L, 4, 9), (5, L, 4, 9)], chunk_size=4, dims_per_head=32, do_pr=True)
    #res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    #check = torch.load(res,map_location='cpu')
    #model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = True
    conf.optim = 'shampoo'
    conf.lr_sched.lr = 1e-4
    wt = Trainer(conf=conf, batch_size=16, model=model)
    wt.setup()
    wt.train()



@launch(nodes={"miramar": 5}, port=29502, start_method="fork", log=True,clear_cache=False)
def Nov14_neotest2_fixedelev():

    import os, torch
    print("yooooo", os.environ.get('LOCAL_RANK'))
    #os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    from train import HresModel, Trainer
    #dataset = HresDataset(batch_size=40) # 650
    # factor, L, n_trans, wsize
    #model = HresModel(steps=[(30, 64, 4, 5)])
    L = 256
    L = 320
    model = HresModel(steps=[(2, L, 4, 9), (3, L, 4, 9), (5, L, 4, 9)], chunk_size=6)
    #res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    res = "/huge/deep/runs_hres/run_Nov14-neotest2-multideltas_20241117-154744/model_step19000_loss0.362.pt"
    check = torch.load(res,map_location='cpu')
    model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    conf.optim = 'shampoo'
    conf.lr_sched.lr = 0.5e-4
    conf.lr_sched.warmup_end_step = 2_000
    conf.lr_sched.cosine_period = 100_000
    wt = Trainer(conf=conf, batch_size=36, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"singing": 3}, port=29501, start_method="spawn", log=True,clear_cache=False)
def Nov14_neotest2_bigbatch():
    dataset = HresDataset(batch_size=24) # 650
    # factor, L, n_trans, wsize
    #model = HresModel(steps=[(30, 64, 4, 5)])
    model = HresModel(steps=[(2, 256, 4, 9), (3, 256, 4, 9), (5, 256, 4, 9)], chunk_size=8)
    #res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    #check = torch.load(res,map_location='cpu')
    #model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"singing": 3}, port=29500, start_method="spawn", log=True,clear_cache=False)
def Nov14_neotest2():
    dataset = HresDataset(batch_size=12) # 650
    # factor, L, n_trans, wsize
    #model = HresModel(steps=[(30, 64, 4, 5)])
    model = HresModel(steps=[(2, 256, 4, 9), (3, 256, 4, 9), (5, 256, 4, 9)])
    #res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    #check = torch.load(res,map_location='cpu')
    #model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


if __name__ == '__main__':
    run(locals().values())
