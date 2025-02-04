import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model_latlon.data import get_constant_vars
from utils import * 
import itertools
from functools import reduce
from model_latlon.primatives2d import *
from model_latlon.harebrained2d import *
from model_latlon.harebrained3d import *
from model_latlon.primatives3d import *
from model_latlon.top import *
from model_latlon.decoder import *
from datasets import *
# to run: pytest -v -s ./model_latlon/test.py

def _test_updown(H,W,kernel_size=5):
    x = torch.ones(1,1,H,W)
    down1 = weights1(EarthConvDown2d(1,2,kernel_size=kernel_size))
    down2 = weights1(EarthConvDown2d(2,4,kernel_size=kernel_size))
    up2 = weights1(EarthConvUp2d(4,2,kernel_size=kernel_size))
    up1 = weights1(EarthConvUp2d(2,1,kernel_size=kernel_size))

    x = down1(x)
    assert np.all(find_periodicity(x) == 1), find_periodicity(x)
    x = down2(x)
    assert np.all(find_periodicity(x) == 1), find_periodicity(x)
    x = up2(x)
    assert np.all(find_periodicity(x) == 2), find_periodicity(x)
    x = up1(x)     
    assert np.all(find_periodicity(x) == 4), find_periodicity(x)

    assert x.shape == (1,1,H,W)

    print("Up Down test done")

def test_updown():
    _test_updown(361,720,kernel_size=3)
    return
    _test_updown(361,720,kernel_size=5)
    _test_updown(361,720,kernel_size=7)
    _test_updown(360,720,kernel_size=3)
    _test_updown(360,720,kernel_size=5)
    _test_updown(360,720,kernel_size=7)


def test_basic_harebrain():
    x = torch.ones(1,1,91,180)
    to = weights1(ToHarebrained2d(1,1))
    fro = weights1(FromHarebrained2d(1,1))
    pad = weights1(HarebrainedPad2d(1,kernel_height=3))
    conv = weights1(nn.Conv2d(1,1,kernel_size=[3,3]))
    x = to(x)
    for xx in x:
        assert np.all(find_periodicity(xx) == 1)
    x = pad(x)
    #for xx in x:
    #    print(find_periodicity(xx))
    #    print(".")
    x = [conv(xx) for xx in x]
    for xx in x:
        p = find_periodicity(xx)
        assert np.max(p) <= 2, p
        assert np.min(p) == 1

    x = fro(x)
    p = find_periodicity(x)
    assert np.all(np.unique(p) == [1,2,4])
    assert np.median(p) in [1,2], p
    print("Done hest_basic_harebrain")

def test_5_harebrain():
    x = torch.ones(1,1,91,180)
    to = weights1(ToHarebrained2d(1,1))
    fro = weights1(FromHarebrained2d(1,1))
    pad = weights1(HarebrainedPad2d(1,kernel_height=5))
    conv = weights1(nn.Conv2d(1,1,kernel_size=[5,5]))
    x = to(x)
    for xx in x:
        assert np.all(find_periodicity(xx) == 1)
    x = pad(x)
    #for xx in x:
    #    print(find_periodicity(xx))
    #    print(".")
    x = [conv(xx) for xx in x]
    for xx in x:
        p = find_periodicity(xx)
        assert np.max(p) <= 3, p
        assert np.min(p) == 1, p

    x = fro(x)
    p = find_periodicity(x)
    assert np.all(np.unique(p) == [1,2,4])
    assert np.median(p) in [1,2], p
    print("Done")

def test_hb_res():
    x = torch.ones(1,1,181,360)
    to = weights1(ToHarebrained2d(1,1))
    res1 = weights1(HarebrainedResBlock2d(1,1,group_norms=[None,None]))
    fro = weights1(FromHarebrained2d(1,1))

    x = to(x)
    x = res1(x)
    x = fro(x)     

    print("done")


def test_earth_res_3d():
    x = torch.ones(1,1,4,91,180)
    res1 = weights1(EarthResBlock3d(1,1))
    x = res1(x)
    assert x.shape == (1,1,4,91,180)
    p = find_periodicity_3d(x)
    assert np.all(p == 1)
    #imsave3('hb/x.png',x[:,:,0])
    print("done")

def test_earth_down_3d():
    x = torch.ones(1,1,4,91,180)
    down = weights1(EarthConvDown3d(1,2))
    x = down(x)
    #imsave3('hb/x.png',x)
    assert x.shape == (1,2,2,46,90), x.shape
    p = find_periodicity_3d(x)
    assert np.all(p == 1), p
    print("done")

def test_earth_updown_3d():
    x = torch.ones(1,1,4,181,360)
    down = weights1(EarthConvDown3d(1,2))
    up = weights1(EarthConvUp3d(2,1))
    x = down(x)
    assert x.shape == (1,2,2,91,180), x.shape
    p = find_periodicity_3d(x)
    assert np.all(p == 1), p
    x = up(x)
    p = find_periodicity_3d(x)
    assert np.all(p == 2), p
    assert x.shape == (1,1,4,181,360), x.shape

    print("done")

def test_earth_updown_2d():
    x = torch.ones(1,1,181,360)
    down = weights1(EarthConvDown2d(1,2,kernel_size=5,stride=2))
    up = weights1(EarthConvUp2d(2,1,kernel_size=5,stride=2))
    x = down(x)
    assert x.shape == (1,2,91,180), x.shape
    x = up(x)
    assert x.shape == (1,1,181,360), x.shape

def test_hb_3d():
    x = torch.ones(1,1,4,181,360)
    to = weights1(ToHarebrained3d(1,1))
    fro = weights1(FromHarebrained3d(1,1))
    x = to(x)
    assert get_center(x).shape[-1] == 360
    for xx in x:
        assert np.all(find_periodicity_3d(xx) == 1), find_periodicity_3d(xx)
    x = fro(x)
    assert x.shape == (1,1,4,181,360)
    p = find_periodicity_3d(x)
    assert np.all(np.unique(p) == [1,2,4])
    assert np.median(p) in [1,2], p
    print("done")

def test_hb_pad_conv_3d():
    x = torch.ones(1,1,4,181,360)
    to = weights1(ToHarebrained3d(1,1))
    pad = weights1(HarebrainedPad3d(1,kernel_height=5))
    conv = weights1(nn.Conv3d(1,1,kernel_size=[1,5,5]))
    fro = weights1(FromHarebrained3d(1,1))
    x = to(x)
    x = pad(x)
    x = [conv(xx) for xx in x]
    x = fro(x)
    assert x.shape == (1,1,4,181,360)
    p = find_periodicity_3d(x)
    assert np.all(np.unique(p) == [1,2,4])
    assert np.median(p) in [1,2], p
    print("done")


def test_full_encoder_decoder():    
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    fe = ResConvEncoder(mesh,ForecastModelConfig(mesh)).to('cuda')
    fd = ResConvDecoder(mesh,ForecastModelConfig(mesh)).to('cuda')
    print_total_params(fe)
    x = torch.ones(1,720,1440,len(mesh.full_varlist)).half().to('cuda')
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        y = fe(x,torch.tensor([0]).to('cuda'))
        y = fd(y)
    assert y.shape == x.shape, f"{y.shape} != {x.shape}"
    print("done")

def disabled_test_forecast_model():    
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_medium)
    fm = ForecastModel(mesh,ForecastModelConfig(mesh)).to('cuda')
    print_total_params(fm)
    x = torch.ones(1,720,1440,len(mesh.full_varlist)).half().to('cuda')
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        y = fm([x,torch.tensor([0]).to('cuda')],[0])
    assert y[0].shape == x.shape, f"{y.shape} != {x.shape}"
    print("done")

def disabled_test_bachelor_like():
    extra = ['logtp', '15_msnswrf', '45_tcc']
    mesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra, input_levels=levels_medium, levels=levels_joank)
    conf = ForecastModelConfig(mesh,encdec_tr_depth=4,latent_size=896)
    fm = ForecastModel(mesh,conf).to('cuda')
    print_total_params(fm)
    x = torch.ones(1,720,1440,len(mesh.full_varlist)).half().to('cuda')
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        y = fm([x,torch.tensor([0]).to('cuda')],[0])
    assert y[0].shape == x.shape, f"{y.shape} != {x.shape}"
    print("done")

def test_simple_conv_plus():
    device = 'cuda'
    mesh = meshes.LatLonGrid(source='era5-28', input_levels=levels_medium, levels=levels_ecm1)
    conf = ForecastModelConfig(mesh,encdec_tr_depth=2,latent_size=1280)
    decoder = SimpleConvPlusDecoder(mesh,conf).to(device)
    x = torch.zeros(1,5,90,186,1280).to(device)
    x = decoder(x)
    print("done")

def assert_helper(a, b, a_name, b_name):
    tolerance = 0.01 # 0.01% tolerance
    
    # Handle when we have torch.Size objects
    if isinstance(a, torch.Size):
        if a == b:
            print(GREEN(f"\n{a_name} matches {b_name}"))
            return
        else:
            raise AssertionError(f"{a_name} does not match {b_name}")
    
    diff_sum = torch.sum(torch.abs(a - b))
    total = torch.sum(torch.abs(a) + torch.abs(b))
    
    percentage_diff = 100 * diff_sum / total
    if percentage_diff < tolerance:
        print(GREEN(f"\n{a_name} matches {b_name} with tolerance: \n {percentage_diff}"))
    else:
        print(RED(f"percentage diff: {percentage_diff} | total: {total} | diff_sum: {diff_sum}"))
        raise AssertionError(f"{a_name} does not match {b_name}, percentage difference: {percentage_diff}%")
   
    max_elem_difference = torch.max(torch.abs(a - b))
    mean_elem = torch.mean(torch.cat((torch.abs(a.flatten()), torch.abs(b.flatten()))))
    if max_elem_difference / mean_elem < tolerance:
        print(GREEN(f"\n{a_name} matches {b_name} for max elem difference with tolerance: \n {max_elem_difference / mean_elem}"))
    else:
        print(RED(f"max elem diff: {max_elem_difference} | max elem: {mean_elem}"))
        raise AssertionError(f"{a_name} does not match {b_name}, max element difference: {max_elem_difference}")
   
# 1/3/2024
# This test no longer works because we are using the new variable weight functionality
# Keeping this test here for future weight tests, but it will need to be rewritten since the current baseline is what is now currently implemented
def test_old_get_weights():
    mesh_slate = [
        meshes.LatLonGrid(source='era5-28', extra_sfc_vars=['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h', 'tc-maxws', 'tc-minp'], input_levels=levels_medium, levels=levels_ecm1),
        meshes.LatLonGrid(source='era5-28', input_levels=levels_medium, levels=levels_ecm1),
        meshes.LatLonGrid(source='era5-28', input_levels=levels_medium, levels=levels_medium),
        meshes.LatLonGrid(source='era5-16', input_levels=levels_ecm1, levels=levels_ecm1),
        meshes.LatLonGrid(source='era5-28', extra_sfc_pad=2, input_levels=levels_medium, levels=levels_ecm1),
        meshes.LatLonGrid(source='era5-28', extra_sfc_vars=['45_tcc', '168_2d', '246_100u', '247_100v'], input_levels=levels_medium, levels=levels_medium),
        meshes.LatLonGrid(source='era5-28', extra_sfc_vars=['45_tcc', '168_2d', '246_100u', '247_100v'], input_levels=levels_medium, levels=levels_ecm1),
        meshes.LatLonGrid(source='era5-28', extra_sfc_vars=['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h'], input_levels=levels_medium, levels=levels_ecm1),
        meshes.LatLonGrid(source='neohres-20', input_levels=levels_hres, levels=levels_joank),
        meshes.LatLonGrid(source='era5-28', input_levels=levels_medium, levels=levels_joank),
    ]
    
    test_strings = [
        'nan extra vars',
        'levels',
        'levels',
        'input levels',
        'sfc_pad',
        'sfc_vars',
        'sfc_vars + levels',
        'tonna output',
        'source change',
        'levels',
    ]
    
    assert len(test_strings) == len(mesh_slate), "Make sure you have enough test string descriptions for the number of meshes"
    for test_string, mesh in zip(test_strings, mesh_slate):
        print(ORANGE(f"Starting test for {test_string}..."))
        weight_eps = 0.01
        
        def old_load_matrices(mesh):
            with open(f"{CONSTS_PATH}/tc_variable_weights_28.pickle", "rb") as f:
                variable_weights = pickle.load(f)
                
            state_norm, state_norm_matrix = load_state_norm(mesh.wh_lev, mesh)
            # nan_mask is only False for nan values eg. all land for sstk (like the last few rows)
            with open(f"{CONSTS_PATH}/nan_mask.pkl", 'rb') as f:
                nan_mask_dict = pickle.load(f)
            nan_mask = np.ones((len(mesh.lats), len(mesh.lons), mesh.n_vars), dtype=bool)
            
            for i, var_name in enumerate(mesh.sfc_vars):
                if var_name in nan_mask_dict:
                    nan_mask[:,:,mesh.n_pr + i] = nan_mask_dict[var_name]
            
            return variable_weights, nan_mask, state_norm, state_norm_matrix
        
        def old_get_weights(mesh, variable_weights, nan_mask, state_norm, weight_eps):
            # Pasted in from old LatLonGrid
            Lons, Lats = np.meshgrid(mesh.lons, mesh.lats)
            mesh.weights = np.cos(Lats * np.pi/180)
            
            B = 1
            N1 = len(mesh.lats)
            N2 = len(mesh.lons)
            D = mesh.n_vars
            pressure_vars = mesh.pressure_vars
            sfc_vars = mesh.sfc_vars
            
            ohp = np.zeros((mesh.n_pr_vars, mesh.n_levels))
            for iv, v in enumerate(pressure_vars):
                for i, lev in enumerate(mesh.levels):
                    ohp[iv, i] = variable_weights[v][i]
            ohp = ohp.flatten()
            for v in sfc_vars:
                if v not in variable_weights:
                    print("Achtung!!!! Variable weights for %s were not found and set to 2. Joan took a look and thinks it's prolly ok"%v, only_rank_0=True)
            ohp = np.concatenate([ohp, [variable_weights.get(v,2)*0.2 for v in sfc_vars]])
            ohp = 1/ohp
            
            weight_ext = np.ones([B,N1,N2,D])
            weight_ext *= mesh.weights[np.newaxis, :, :, np.newaxis]
            weight_flat = weight_ext.reshape((weight_ext.shape[0], weight_ext.shape[1]*weight_ext.shape[2], weight_ext.shape[3]))
            weight_flat = weight_flat[:, :, :mesh.n_pr]
            weight_flat = weight_flat.reshape((weight_ext.shape[0], weight_ext.shape[1]*weight_ext.shape[2], mesh.n_pr_vars, mesh.n_levels))
            #F = torch.HalfTensor if self.conf.HALF else torch.FloatTensor
            F = torch.FloatTensor
            weight_og = F(mesh.weights) + weight_eps
            mw2 = np.ones(list(mesh.weights.shape) + list(ohp.shape)).astype(np.float32)
            mw2 *= mesh.weights[:,:, np.newaxis]
            mw2 *= ohp[np.newaxis, np.newaxis, :]
            mw2 = np.where(nan_mask, mw2, 0)

            weight = F(mw2)
            sum_weight = torch.sum(weight)

            unnorm_fac = np.zeros((len(pressure_vars), mesh.n_levels))
            dimprint('[unnorm_fac]',unnorm_fac.shape)
            for i, var in enumerate(pressure_vars):
                unnorm_fac[i,:] = state_norm[var][1][mesh.wh_lev]
                
            return weight, weight_og, sum_weight, ohp
        
        # Gather old weights
        _TEMP_variable_weights, _TEMP_nan_mask, _TEMP_state_norm, _ = old_load_matrices(mesh)
        old_total_weight, old_weight_og, old_sum_weight, old_variable_weights = old_get_weights(mesh, _TEMP_variable_weights, _TEMP_nan_mask, _TEMP_state_norm, weight_eps=weight_eps)
        
        # Gather new weights
        from model_latlon.decoder import gather_variable_weights, gather_geospatial_weights
        new_variable_weights = gather_variable_weights(mesh)
        new_geospatial_weights = gather_geospatial_weights(mesh)
        # Taken from compute_loss
        new_total_weight = new_geospatial_weights * new_variable_weights
        new_sum_weight = torch.sum(new_total_weight)
        new_weight_og = new_geospatial_weights + weight_eps
        
        assert_helper(old_total_weight, new_total_weight, "old_total_weight", "new_total_weight")
        assert_helper(old_sum_weight, new_sum_weight, "old_sum_weight", "new_sum_weight")
        assert_helper(old_weight_og.unsqueeze(0), new_weight_og.squeeze(-1), "old_weight_og", "new_weight_og")
        assert_helper(torch.tensor(old_variable_weights[np.newaxis, np.newaxis, np.newaxis, :]), new_variable_weights, "old_variable_weights", "new_variable_weights")
        
        print(GREEN(f"Finished test for {test_string}... \n\n\n"))

def test_check_for_dates():    
    def old_check_for_dates(requested_dates, inputs, outputs, only_at_z, timesteps):
        dates = np.array([to_unix(d) for d in requested_dates])
        datasets = inputs + outputs

        if only_at_z is not None:
            oazp = set(only_at_z)
            for x in only_at_z:
                for dh in config.timesteps:
                    oazp.add((x+dh)%24)
            oazp = sorted(list(oazp))
        else:
            oazp = range(0, 24, 3)
            
        sample_descriptor = [(0,inputs)] + [(t,outputs) for t in timesteps]
        toffs = [0]*len(inputs) + list(itertools.chain.from_iterable([[dt*3600]*len(outputs) for dt in timesteps]))
        print("toffs", toffs)
        samplepos2mesh = list(range(len(inputs)))+[len(inputs)+i for i in range(len(outputs))]*len(timesteps)
        print("samplepos2mesh", samplepos2mesh)
        
        datemin = get_date(np.min(dates)); datemax = get_date(np.max(dates))
        all_i_could_want = np.array([dates+3600*h for h in oazp],dtype=np.int64).flatten()

        instance_times = []
        for i,dataset in enumerate(datasets):
            times = np.intersect1d(all_i_could_want,dataset.get_loadable_times(datemin,datemax))
            instance_times.append(times)
            print(f"📅 {dataset.mesh.string_id:<30} num: {len(times):<5} from {get_date_str(times[0]):<10} to {get_date_str(times[-1]):<10}, min dh: {np.diff(times).min()//3600}hr, max dh: {np.diff(times).max()//3600}hr, mean dh: {np.diff(times).mean()/3600}hr",only_rank_0=True)
            dataset.instance_times = times 
            
        num_instances_per_sample = len(toffs)
        alltimes = [instance_times[samplepos2mesh[i]] - toffs[i] for i in range(len(toffs)) if datasets[samplepos2mesh[i]].is_required]
        sample_times = reduce(np.intersect1d,alltimes)
        assert len(sample_times) > 0, f"No data found for {requested_dates[0]} to {requested_dates[-1]}"

        sample_array = np.zeros((num_instances_per_sample,len(sample_times)),dtype=np.int64)
        for i in range(num_instances_per_sample):
            sample_array[i] = sample_times + toffs[i]
            assert np.all(np.isin(sample_array[i],instance_times[samplepos2mesh[i]])) or not datasets[samplepos2mesh[i]].is_required, f"Something is fucked"
        sample_array = sample_array.T
        assert sample_array.shape[1] == len(samplepos2mesh), f"{sample_array.shape} vs {samplepos2mesh}"
        sample_array = sample_array 
        print(f"📅📅 Total Num Dates: {sample_array.shape[0]}, from {get_date_str(sample_array[0][0].item())} to{get_date_str(sample_array[-1][0].item())}",only_rank_0=True)
        print(f"📅📅 Total Min dh {np.diff(sample_array[:,0]).min() // 3600}hr, Max dh {np.diff(sample_array[:,0]).max() // 3600}hr, Mean dh {np.diff(sample_array[:,0]).mean() / 3600}hr",only_rank_0=True)

        return sample_descriptor, toffs, samplepos2mesh, instance_times, alltimes, sample_times, sample_array

    def new_check_for_dates(requested_dates, inputs, outputs, only_at_z, timesteps):
        unix_dates = np.array([to_unix(d) for d in requested_dates])
        datasets = inputs + outputs

        # Only gather times at these z values 
        only_at_z = range(0, 24, 3)
        if only_at_z is not None:
            only_at_z = set(only_at_z)
            
            # Make sure we also have the timesteps
            for z, timestep in zip(only_at_z, timesteps):
                additional_z = ( z + timestep ) % 24
                only_at_z.add(additional_z)
                
            only_at_z = sorted(list(only_at_z))
            
        sample_descriptor =  [(0, inputs)] # Inputs
        sample_descriptor += [(t, outputs) for t in timesteps] # Outputs
        
        instance_unix_time_offsets =  [0] * len(inputs)
        instance_unix_time_offsets += list(itertools.chain.from_iterable([[dt*3600]*len(outputs) for dt in timesteps]))
        print("Instance unix time offsets: ", instance_unix_time_offsets)
        
        instance_position_to_dataset_position =  list(range(len(inputs)))
        instance_position_to_dataset_position += [len(inputs)+i for i in range(len(outputs))]*len(timesteps)
        print("Instance position to dataset position: ", instance_position_to_dataset_position)
        
        # This assert should be theoretically impossible to break given the code above, but in either case it should be true
        assert len(instance_unix_time_offsets) == len(instance_position_to_dataset_position), "Length of instance_unix_time_offsets and instance_position_to_dataset_position must be equal. Not a consistent number of instances represented in the sample"
        
        unix_datemin = get_date(np.min(unix_dates))
        unix_datemax = get_date(np.max(unix_dates))
        
        # All unix times that we could want
        all_unix_times_i_could_want = np.array([unix_dates + 3600*h for h in only_at_z], dtype=np.int64).flatten()
        
        dataset_unix_times = []
        for dataset in datasets:
            # all_unix_times_i_could_want represents the total number of dates available from model config
            # dataset.get_loadable_times(unix_datemin, unix_datemax) represents the total number of dates available for the specific dataset type
            shared_unix_times = np.intersect1d(all_unix_times_i_could_want, dataset.get_loadable_times(unix_datemin, unix_datemax))
            
            dataset_unix_times.append(shared_unix_times)
            print(f"📅 {dataset.mesh.string_id:<30} num: {len(shared_unix_times):<5} from {get_date_str(shared_unix_times[0]):<10} to {get_date_str(shared_unix_times[-1]):<10}, min dh: {np.diff(shared_unix_times).min()//3600}hr, max dh: {np.diff(shared_unix_times).max()//3600}hr, mean dh: {np.diff(shared_unix_times).mean()/3600}hr",only_rank_0=True)
            
        # Normalizes dataset unix dates across instances
        num_instances = len(instance_unix_time_offsets)
        gathered_unix_times = [
            dataset_unix_times[instance_position_to_dataset_position[i]] - instance_unix_time_offsets[i]
            for i in range(num_instances) 
            if datasets[instance_position_to_dataset_position[i]].is_required
        ]
        sample_unix_times = reduce(np.intersect1d, gathered_unix_times)
        assert len(sample_unix_times) > 0, f"No data found for {requested_dates[0]} to {requested_dates[-1]}"
        
        sample_array = np.zeros((num_instances,len(sample_unix_times)),dtype=np.int64)
        
        for i in range(num_instances):
            sample_array[i] = sample_unix_times + instance_unix_time_offsets[i]
            assert np.all(np.isin(sample_array[i], dataset_unix_times[instance_position_to_dataset_position[i]])) or not datasets[instance_position_to_dataset_position[i]].is_required, f"Something is fucked"
        
        sample_array = sample_array.T
        assert sample_array.shape[1] == len(instance_position_to_dataset_position), f"{sample_array.shape} vs {instance_position_to_dataset_position}"
        sample_array = sample_array 
        print(f"📅📅 Total Num Dates: {sample_array.shape[0]}, from {get_date_str(sample_array[0][0].item())} to{get_date_str(sample_array[-1][0].item())}",only_rank_0=True)
        print(f"📅📅 Total Min dh {np.diff(sample_array[:,0]).min() // 3600}hr, Max dh {np.diff(sample_array[:,0]).max() // 3600}hr, Mean dh {np.diff(sample_array[:,0]).mean() / 3600}hr",only_rank_0=True)
        
        return sample_descriptor, instance_unix_time_offsets, instance_position_to_dataset_position, dataset_unix_times, gathered_unix_times, sample_unix_times, sample_array
        
    requested_dates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2023, 12, 21))])
    inputs = [AnalysisDataset(meshes.LatLonGrid(source='era5-28',input_levels=levels_medium, levels=levels_ecm1))]
    outputs = [AnalysisDataset(meshes.LatLonGrid(source='era5-28',input_levels=levels_medium, levels=levels_ecm1))]
    only_at_z = None
    timesteps = [0,6,24,96,144]
    start_time = time.time()
    old1, old2, old3, old4, old5, old6, old7 = old_check_for_dates(requested_dates, inputs, outputs, only_at_z, timesteps)
    print("took: ", time.time() - start_time)
    start_time = time.time()
    new1, new2, new3, new4, new5, new6, new7 = new_check_for_dates(requested_dates, inputs, outputs, only_at_z, timesteps)
    print("new took: ", time.time() - start_time)
    print(old1)
    print(new1)
    print('')
    print(old2)
    print(new2)
    print('')
    print(old3)
    print(new3)
    print('')
    print(old4)
    print(new4)
    print('')
    print(old5)
    print(new5)
    print('')
    print(old6)
    print(new6)
    print('')
    print(old7)
    print(new7)

if __name__ == "__main__":
    #test_full_encoder_decoder()
    #test_simple_conv_plus()
    #test_old_get_weights()
    test_check_for_dates()