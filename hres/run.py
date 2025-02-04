import sys
sys.path.append("../runs")
sys.path.append("..")
from launch import *
from train import *

@launch(nodes={"bimini": 6}, port=29500, start_method="spawn", log=False)
def Apr18_doot_silly():
    dataset = HresDataset(batch_size=512)
    model = HresModel(silly_rel=True)
    conf.lr_sched.lr = 2e-4
    #conf.nope = True
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"bimini": 6}, port=29500, start_method="spawn", log=False)
def Apr18_doot_highlr():
    dataset = HresDataset(batch_size=512)
    model = HresModel()
    conf.lr_sched.lr = 2e-3
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"bimini": 6}, port=29500, start_method="spawn", log=False)
def Apr22_doot_bigstate():
    dataset = HresDataset(batch_size=330)
    model = HresModel(L=1024)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"martins": 1}, port=29500, start_method="spawn", log=False)
def Apr22_doot_abscrap():
    dataset = HresDataset(batch_size=512)
    model = HresModel(L=128)
    conf.lr_sched.lr = 2e-5
    conf.nope = True
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"miramar": 6}, port=29500, start_method="spawn", log=False)
def Apr18_doot_lowlr():
    dataset = HresDataset(batch_size=512)
    model = HresModel()
    conf.lr_sched.lr = 2e-5
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"halfmoon": 4}, port=29500, start_method="spawn", log=False)
def Apr22_doot_flash():
    dataset = HresDataset(batch_size=512)
    model = HresModel()
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"halfmoon": 4}, port=29500, start_method="spawn", log=False)
def Apr22_doot_bugfix():
    dataset = HresDataset(batch_size=512)
    model = HresModel()
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"singing": 4}, port=29500, start_method="spawn", log=False)
def Apr22_doot_bugfix_rad():
    dataset = HresDataset(batch_size=512)
    model = HresModel(do_radiation=True)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"stinson": 4}, port=29500, start_method="spawn", log=False)
def Apr22_doot_bugfix_modis():
    dataset = HresDataset(batch_size=512)
    model = HresModel(do_modis=True)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"stinson": 4}, port=29500, start_method="spawn", log=True)
def Jun4_whatever():
    dataset = HresDataset(batch_size=300)
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True)
    #res = "/fast/ignored/runs_hres/run_May5-multivar_20240514-123630/model_step34000_loss0.168.pt"
    #check = torch.load(res,map_location='cpu')
    #model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = True
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"singing": 1}, port=29500, start_method="spawn", log=True)
def Sep23_test():
    dataset = HresDataset(batch_size=384) # 650
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, grid="_small")
    conf.nope = True
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"baga.fast": 4, "singing": 4}, port=29500, start_method="spawn", log=True)
def Sep24_oldnewslop():
    dataset = HresDataset(batch_size=650) # 650
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, grid="_small")
    res = "/huge/deep/runs_hres/run_Aug23-neoslop_20240828-091119/model_step34500_loss0.391.pt"
    res = "/huge/deep/runs_hres/run_Aug23-neoslop_20240829-073632/model_step331500_loss0.363.pt"
    res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240924-135233/model_step8000_loss0.253.pt"
    res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240925-021745/model_step72500_loss0.227.pt"
    res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240926-200416/model_step52500_loss0.248.pt"
    rse = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    check = torch.load(res,map_location='cpu')
    model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"glass": 6}, port=29500, start_method="spawn", log=True)
def Oct28_coolerpacific():
    dataset = HresDataset(batch_size=650) # 650
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, grid="_small")
    # res = "/huge/deep/runs_hres/run_Aug23-neoslop_20240828-091119/model_step34500_loss0.391.pt"
    # check = torch.load(res,map_location='cpu')
    # model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"baga.fast": 4, "singing": 4}, port=29500, start_method="spawn", log=True)
def Oct31_retro():
    dataset = HresDataset(batch_size=650) # 650
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, grid="_small")
    # res = "/huge/deep/runs_hres/run_Sep24-oldnewslop_20240930-143452/model_step495000_loss0.193.pt"
    # check = torch.load(res,map_location='cpu')
    # model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

#@launch(nodes={"miramar.fast": 6, "barceloneta": 6}, port=29500, start_method="spawn", log=True)
@launch(nodes={"stinson": 4, "singing.fast": 4}, port=29500, start_method="spawn", log=True)
def Aug23_neoslop():
    dataset = HresDataset(batch_size=650) # 650
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True, grid="_small")
    res = "/fast/ignored/runs_hres/run_Jun14-sloptest-small_20240615-231445/model_step111000_loss0.430.pt"
    res = "/fast/ignored/runs_hres/run_Jun14-sloptest-small_20240618-194020/model_step75000_loss0.377.pt"
    res = "/fast/ignored/runs_hres/run_Jun14-sloptest-small_20240620-171946/model_step169000_loss0.366.pt"
    res = "/huge/deep/runs_hres/run_Aug23-neoslop_20240823-170423/model_step71500_loss0.403.pt"
    res = "/huge/deep/runs_hres/run_Aug23-neoslop_20240825-210103/model_step64000_loss0.354.pt"
    res = "/huge/deep/runs_hres/run_Aug23-neoslop_20240827-115630/model_step8000_loss0.338.pt"
    res = "/huge/deep/runs_hres/run_Aug23-neoslop_20240827-165139/model_step3000_loss0.337.pt"
    res = "/huge/deep/runs_hres/run_Aug23-neoslop_20240827-213146/model_step500_loss0.394.pt"
    res = "/huge/deep/runs_hres/run_Aug23-neoslop_20240828-091119/model_step34500_loss0.391.pt"
    check = torch.load(res,map_location='cpu')
    model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"bimini.fast": 6, "barceloneta": 6}, port=29500, start_method="spawn", log=True)
def Jun14_sloptest():
    dataset = HresDataset(batch_size=320)
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True)
    res = "/fast/ignored/runs_hres/run_Jun14-sloptest_20240614-205902/model_step5000_loss0.533.pt"
    res = "/fast/ignored/runs_hres/run_Jun14-sloptest_20240615-172518/model_step10500_loss0.578.pt"
    res = "/fast/ignored/runs_hres/run_Jun14-sloptest_20240615-231544/model_step109500_loss0.431.pt"
    res = "/fast/ignored/runs_hres/run_Jun14-sloptest_20240618-193944/model_step29000_loss0.328.pt"
    res = "/fast/ignored/runs_hres/run_Jun14-sloptest_20240620-172112/model_step169000_loss0.376.pt"
    check = torch.load(res,map_location='cpu')
    model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()



@launch(nodes={"bimini.fast": 4, "singing.fast": 4, "stinson.fast": 4, "halfmoon.fast": 4, "miramar": 4}, port=29500, start_method="spawn", log=True)
def May5_multivar():
    dataset = HresDataset(batch_size=300)
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True)
    res = "/fast/ignored/runs_hres/run_May5-multivar_20240506-141104/model_step10500_loss0.302.pt"
    res = "/fast/ignored/runs_hres/run_May5-multivar_20240506-191828/model_step39000_loss0.251.pt"
    res = "/fast/ignored/runs_hres/run_May5-multivar_20240507-175311/model_step23000_loss0.179.pt"
    res = "/fast/ignored/runs_hres/run_May5-multivar_20240508-103121/model_step61000_loss0.160.pt"
    res = "/fast/ignored/runs_hres/run_May5-multivar_20240510-144336/model_step169000_loss0.157.pt"
    res = "/fast/ignored/runs_hres/run_May5-multivar_20240514-123630/model_step34000_loss0.168.pt"
    check = torch.load(res,map_location='cpu')
    model.load_state_dict(check['model_state_dict'],strict=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"bimini.fast": 4, "singing.fast": 4, "stinson.fast": 4, "halfmoon.fast": 4, "miramar": 4}, port=29500, start_method="spawn", log=True)
def May3_neowallstick():
    dataset = HresDataset(batch_size=300)
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"bimini.fast": 6, "miramar": 6}, port=29500, start_method="spawn", log=False)
def Apr25_wallstick():
    dataset = HresDataset(batch_size=290)
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"singing.fast": 4, "stinson": 4}, port=29500, start_method="spawn", log=False)
def Apr25_wallstick_fp32():
    dataset = HresDataset(batch_size=162)
    model = HresModel(do_modis=True, do_radiation=True, do_pressure=True, L=768, absbias=True, depth=16, silly_rel=True)
    conf.nope = False
    conf.HALF = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"miramar": 6}, port=29500, start_method="spawn", log=False)
def Apr22_doot_mega():
    dataset = HresDataset(batch_size=256)
    model = HresModel(depth=16, L=768, do_pressure=True, absbias=True, silly_rel=True, use_matepoint=False)
    conf.nope = True
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"stinson": 4}, port=29500, start_method="spawn", log=False)
def Apr22_doot_bugfix_absbias():
    dataset = HresDataset(batch_size=512)
    model = HresModel(absbias=True)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

@launch(nodes={"stinson": 4}, port=29500, start_method="spawn", log=False)
def Apr22_doot_bugfix_long():
    dataset = HresDataset(batch_size=450)
    model = HresModel(absbias=False, depth=20, use_matepoint=False)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"singing": 4}, port=29500, start_method="spawn", log=False)
def Apr22_doot_bugfix_absbias_pr():
    dataset = HresDataset(batch_size=512)
    model = HresModel(absbias=True, do_pressure=True)
    conf.nope = False
    #conf.use_l2 = True
    #conf.initial_gradscale = 2048.
    #conf.clip_gradient_norm = 0.5
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()


@launch(nodes={"singing": 4}, port=29500, start_method="spawn", log=False)
def Apr22_doot_bugfix_absbias_silly():
    dataset = HresDataset(batch_size=512)
    model = HresModel(absbias=True, silly_rel=True)
    conf.nope = False
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()



@launch(nodes={"halfmoon": 4}, port=29500, start_method="spawn", log=False)
def Apr18_doot():
    dataset = HresDataset(batch_size=512)
    model = HresModel()
    wt = Trainer(conf=conf, dataset=dataset, model=model)
    wt.setup()
    wt.train()

if __name__ == '__main__':
    run(locals().values())
