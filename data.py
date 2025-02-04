import torch
import time 
import numpy as np 
from utils import *
import copy
from datetime import datetime, timedelta
import os 
import itertools
import xarray as xr
from meshes import *
from collections import defaultdict


from functools import reduce

from datasets import ObsDataset, AnalysisDataset, StationDataset


class TrainingSample():
    def __init__(self):
        # Format for inputs / outputs is 
        #    a list (# encoders / decoders) 
        #       of two lists (mesh_id, tensors) 
        #           of lists (# of meshes):
        # [
        #   [
        #     [mesh_ids, ...], [tensors, ...]
        #   ],
        #   [
        #     [mesh_ids, ...], [tensors, ...]
        #   ],
        #   ...
        # ]
        self.inputs = []
        self.outputs = []
        self.additional_inputs = defaultdict(list)
        # List of unix timesteps (order matched)
        # First timestep is the input timestep (t0), the rest are output timesteps
        self.timestamps = [] 
    
    def update(self):
        assert len(self.inputs) + len(self.outputs) == len(self.timestamps)
        self.t0 = max(self.timestamps[:len(self.inputs)])
        self.dts = [int((t-self.t0)/3600) for t in self.timestamps[1:]]

    def get_x_t0(self, encoders=None):
        if encoders is not None:
            output_x_t0 = []
            string_ids = self.inputs[0][0]
            for encoder in encoders:
                assert encoder.mesh.string_id in string_ids, f"Encoder {encoder.__class__.__name__}'s mesh ({encoder.mesh.string_id}) not found in sample (input string_ids: {string_ids})"
                
                mesh_idx = string_ids.index(encoder.mesh.string_id)
                output_x_t0.append(self.inputs[0][1][mesh_idx])
            
            # For the moment add timestep to the end (not sure if this is actually necessary, but its convention rn)
            output_x_t0.append(torch.tensor([self.timestamps[0]]))
            return output_x_t0
        
        # Basic use of get_x_t0 (where we assume there is only one encoder, so we can just match it easily)
        else:
            assert len(self.inputs) == 1
            return [self.inputs[0][1][0], torch.tensor([self.timestamps[0]])]

    # Doesn't have a basic usage yet (on purpose, basic usage may want to be its own function)
    # Outputs data in format:
    # [
    #    [decoder1_tensor_at_timestamp1, decoder2_tensor_at_timestamp1, ..., timestamp1],
    #    [decoder1_tensor_at_timestamp2, decoder2_tensor_at_timestamp2, ..., timestamp2],
    #    ...
    # ]
    def get_y_ts(self, decoders):
        output_y_ts = []
        for index_timestamp, timestamp in enumerate(self.timestamps[1:]):
            output_y_ts.append([])
            string_ids = self.outputs[index_timestamp][0]
            for decoder in decoders:
                assert decoder.mesh.string_id in string_ids, f"Decoder {decoder.__class__.__name__}'s mesh ({decoder.mesh.string_id}) not found in sample (output string_ids: {string_ids})"

                mesh_idx = string_ids.index(decoder.mesh.string_id)
                output_y_ts[-1].append(self.outputs[index_timestamp][1][mesh_idx])
            output_y_ts[-1].append(torch.tensor([timestamp]))
        return output_y_ts

    def get_additional_inputs(self):
        return self.additional_inputs

class DataConfig():
    def __init__(self,inputs=[],outputs=[],conf_to_copy=None,**kwargs):
        self.inputs = inputs 
        self.outputs = outputs
        self.timesteps = [24]
        self.requested_dates = get_dates((D(1997, 1, 1),D(2017, 12,1)))
        self.only_at_z = None
        self.clamp = 13
        self.clamp_output=None
        self.random_timestep_subset = False
        self.realtime = False
        self.ens_nums = None
        self.use_rh = False

        if conf_to_copy is not None:
            self.__dict__.update(copy.deepcopy(conf_to_copy.__dict__))
        for k,v in kwargs.items():
            assert hasattr(self,k), f"{k} is not a NeoDataConfig attribute"
            setattr(self,k,v)
            
        for dataset in (self.inputs + self.outputs):
            if dataset.mesh.shape() == -1: print(ORANGE("WARNING: Mesh shape is -1, shape will not be checked (specify mesh shape in meshes.py for your specific mesh type)"))
        self.update()

    def update(self):
        if len(self.outputs) == 0:
            self.outputs = self.inputs
        self.proc_start_t = time.time()

        if self.clamp_output is None:
            self.clamp_output = self.clamp
        ts0 = [0] + self.timesteps
        #assert np.diff(ts0).min() == np.diff(ts0).max()
        assert ts0 == sorted(ts0)
        self.model_dh = np.diff(ts0)[0] if len(ts0) > 1 else None

        if self.realtime:
            for i,mesh in enumerate(self.inputs):
                self.inputs[i] = to_realtime(mesh)
                if self.ens_nums is not None:
                    self.inputs[i].ens_num = self.ens_nums[i]
        else:
            assert self.ens_nums is None, "Ensemble numbers are only supported in realtime mode"

def to_realtime(mesh):
    mesh = copy.deepcopy(mesh) # I think it's best to deepcopy the mesh but tbh not 100% sure
    match mesh.source:
        case 'neogfs-25': mesh.source = 'gfs_rt-25'
        case 'hres-13': 
            mesh.source = 'ens_rt-16'
            mesh.input_levels = levels_ecm1
            mesh.intermediate_levels = [levels_tiny]
    return mesh




def get_sample_idxs(sample_idx,config):
    c = config
    sample_date = c.sample_dates[sample_idx]
    iidx_at_date = lambda idate,sources : [date2idx(idate,s,c) for s in sources]
    sample_idx = [iidx_at_date(sample_date+timedelta(hours=dt),sources) for dt,sources in c.sample_descriptor]
    return sample_idx

def date2idx(date,source,c):
    idx = c.instance_dates_dict[date][source]
    return idx

class WeatherDataset():

    def __init__(self,config):
        self.config = config 
        self.is_first_load = True
        self.load_timer = Timer("load_timer")
        

    def __len__(self):
        return self.sample_array.shape[0]
    
    def __getitem__(self,idx):
        if not hasattr(self,'sample_array'):
            assert False, "You need to call check_for_dates before you can get items"
        ohp = 0
        trys = 0
        while True:
            try:
                s = self.get_sample((idx+ohp) % self.sample_array.shape[0])
                trys = 0; ohp = 0
                return s
            except Exception as e:
                raise e
                if trys > 15: 
                    raise e
                print(f"Error in get_sample: {e}")
                ohp = (ohp + 97) % self.sample_array.shape[0]
                trys += 1

    def unflatten(self,l):
        out = []
        j=0
        for d in self.sample_descriptor:
            out.append(l[j:j+len(d[1])])
            j += len(d[1])
        return out

    def get_sample(self,idx):
        ts = self.sample_array[idx]
        ts = self.unflatten(ts)

        if self.is_first_load:
            print("Loading first sample... ")
        with self.load_timer():
            if self.config.random_timestep_subset:
                ts = [ts[0]] + [ts[i+1] for i in sorted(torch.randperm(len(ts)-2)[:self.config.random_timestep_subset-1])] + [ts[-1]] # the first one is the input, last one we always want cause it's the max
            
            sample = TrainingSample()
            for index_t, times in enumerate(ts):
                instances = [[], []]
                for index_mesh, unix_t in enumerate(times):
                    dataset = self.sample_descriptor[index_t][1][index_mesh]
                    if isinstance(dataset, StationDataset):
                        data = dataset.load_data(unix_t)
                        assert data[0]['data'].shape[-1] == dataset.mesh.shape()[0], f"Last dimension of obs tensor should be equal to the number of variables in the dataset mesh"
                        instances[0].append(dataset.mesh.string_id)
                        instances[1].append(data)
                        sample.additional_inputs["station_data"].append(data)
                        sample.additional_inputs["timestamps"].append(unix_t)
                    else:
                        tensor = self.gather_tensor(unix_t, dataset, index_t > 0)
                        assert isinstance(tensor, torch.Tensor), "You are required to submit the data as a torch tensor" 
                        if dataset.mesh.shape() != -1: assert all([x==y or x==-1 for x, y in zip(dataset.mesh.shape(), tensor.shape)]), f"{dataset.__class__.__name__} expects a tensor of shape: {dataset.mesh.shape()} but received a tensor of shape: {tensor.shape}"
                        instances[0].append(dataset.mesh.string_id)
                        instances[1].append(tensor)
                if index_t == 0: # Note: It's possible that in the future, we will want to have more than one input timestep, like when doing DA stuff. We'll have to update this then.
                    sample.inputs.append(instances)
                else:
                    sample.outputs.append(instances)
                sample.timestamps.append(unix_t)
            sample.update()
                    
        if self.is_first_load:
            self.is_first_load = False
            print(f"First load took {self.load_timer.val:.6f}s")
        return sample

    def gather_tensor(self, unix_t, dataset, is_output=True):

        # TODO: we should stop having these ifs here and instead make the interfaces consistent.

        if isinstance(dataset, TCRegionalIntensities):
            return torch.tensor(get_tcregional_output(get_date(unix_t), dataset), dtype=torch.float16)

        if issubclass(type(dataset), ObsDataset):
            return dataset.load_data(unix_t)
        
        if isinstance(dataset, AnalysisDataset):
            return dataset.load_data(unix_t, is_output)
        
        if isinstance(dataset, StationDataset):
            return dataset.load_data(unix_t)
        
         
    @TIMEIT()
    def check_for_dates(self):
        # Instance = One weather state
        # Sample   = All instances across relevant timesteps 
        unix_dates = np.array([to_unix(d) for d in self.config.requested_dates])
        self.datasets = self.config.inputs + self.config.outputs

        # Only gather times at these z values 
        only_at_z = range(0, 24, 3)
        if self.config.only_at_z is not None:
            only_at_z = set(self.config.only_at_z)
            
            # Make sure we also have the timesteps
            for z, timestep in zip(self.config.only_at_z, self.config.timesteps):
                additional_z = ( z + timestep ) % 24
                only_at_z.add(additional_z)
                
            only_at_z = sorted(list(only_at_z))
            
        self.sample_descriptor =  [(0, self.config.inputs)] # Inputs
        self.sample_descriptor += [(t, self.config.outputs) for t in self.config.timesteps] # Outputs
        
        instance_unix_time_offsets =  [0] * len(self.config.inputs)
        instance_unix_time_offsets += list(itertools.chain.from_iterable([[dt*3600]*len(self.config.outputs) for dt in self.config.timesteps]))
        print("Instance unix time offsets: ", instance_unix_time_offsets)
        
        self.instance_position_to_dataset_position =  list(range(len(self.config.inputs)))
        self.instance_position_to_dataset_position += [len(self.config.inputs)+i for i in range(len(self.config.outputs))]*len(self.config.timesteps)
        print("Instance position to dataset position: ", self.instance_position_to_dataset_position)
        
        # This assert should be theoretically impossible to break given the code above, but in either case it should be true
        assert len(instance_unix_time_offsets) == len(self.instance_position_to_dataset_position), "Length of instance_unix_time_offsets and instance_position_to_dataset_position must be equal. Not a consistent number of instances represented in the sample"
        
        unix_datemin = get_date(np.min(unix_dates))
        unix_datemax = get_date(np.max(unix_dates))
        
        all_unix_times_i_could_want = np.array([unix_dates + 3600*h for h in only_at_z], dtype=np.int64).flatten()
        
        dataset_unix_times = []
        for dataset in self.datasets:
            # all_unix_times_i_could_want represents the total number of dates available from model config
            # dataset.get_loadable_times(unix_datemin, unix_datemax) represents the total number of dates available for the specific dataset type
            shared_unix_times = np.intersect1d(all_unix_times_i_could_want, dataset.get_loadable_times(unix_datemin, unix_datemax))

            if len(shared_unix_times) == 0:
                print("No shared unix times for dataset: ", dataset)
                print("all_unix_times_i_could_want: ", all_unix_times_i_could_want)
                print("dataset.get_loadable_times(unix_datemin, unix_datemax): ", dataset.get_loadable_times(unix_datemin, unix_datemax))
                raise ValueError("No shared unix times for dataset: ", dataset)
            
            dataset_unix_times.append(shared_unix_times)
            print(f"📅 {dataset.mesh.string_id:<30} num: {len(shared_unix_times):<5} from {get_date_str(shared_unix_times[0]):<10} to {get_date_str(shared_unix_times[-1]):<10}, min dh: {np.diff(shared_unix_times).min()//3600}hr, max dh: {np.diff(shared_unix_times).max()//3600}hr, mean dh: {np.diff(shared_unix_times).mean()/3600}hr",only_rank_0=True)
            
        # Normalizes dataset unix dates across instances
        num_instances = len(instance_unix_time_offsets)
        gathered_unix_times = [
            dataset_unix_times[self.instance_position_to_dataset_position[i]] - instance_unix_time_offsets[i]
            for i in range(num_instances) 
            if self.datasets[self.instance_position_to_dataset_position[i]].is_required
        ]
        sample_unix_times = reduce(np.intersect1d, gathered_unix_times)
        assert len(sample_unix_times) > 0, f"No data found for {self.config.requested_dates[0]} to {self.config.requested_dates[-1]}"
        
        sample_array = np.zeros((num_instances,len(sample_unix_times)),dtype=np.int64)
        
        for i in range(num_instances):
            sample_array[i] = sample_unix_times + instance_unix_time_offsets[i]
            assert np.all(np.isin(sample_array[i], dataset_unix_times[self.instance_position_to_dataset_position[i]])) or not self.datasets[self.instance_position_to_dataset_position[i]].is_required, f"Something is fucked"
        
        sample_array = sample_array.T
        assert sample_array.shape[1] == len(self.instance_position_to_dataset_position), f"{sample_array.shape} vs {self.instance_position_to_dataset_position}"
        self.sample_array = sample_array 
        print(f"📅📅 Total Num Dates: {sample_array.shape[0]}, from {get_date_str(sample_array[0][0].item())} to{get_date_str(sample_array[-1][0].item())}",only_rank_0=True)
        print(f"📅📅 Total Min dh {np.diff(sample_array[:,0]).min() // 3600}hr, Max dh {np.diff(sample_array[:,0]).max() // 3600}hr, Mean dh {np.diff(sample_array[:,0]).mean() / 3600}hr",only_rank_0=True)

if __name__ == '__main__' and os.environ.get('JOAN'):
    from utils import get_dates, D, levels_medium
    #from datasets import MicrowaveLoader, IgraLoader
    from model_latlon.da_encoders import MicrowaveData, BalloonData
    atms_mesh = MicrowaveData("1bamua") # atms,1bamua
    igra_mesh = BalloonData()

    import meshes
    extra_all = []
    imesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    imesh2 = atms_mesh#meshes.LatLonGrid(source='neohres-20', extra_sfc_vars=extra_all, input_levels=levels_hres, levels=levels_medium)
    imesh3 = igra_mesh
    omesh = meshes.LatLonGrid(source='era5-28', extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    tdates = get_dates([(D(2010, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])
    
    timesteps = [0, 24]
    data = WeatherDataset(DataConfig(
        inputs=[imesh, imesh2, imesh3], outputs=[omesh],
        timesteps=timesteps,
        requested_dates = tdates,
        ))

    data.check_for_dates()
    #self.data_loader = DataLoader(self.data, batch_size=self.conf.batch_size, shuffle=True, num_workers=2, prefetch_factor=2,worker_init_fn=init_fn)
    from train import collate_fn
    dl = torch.utils.data.DataLoader(data, batch_size=1, num_workers=0,shuffle=True, collate_fn=collate_fn)
    for i, sample in enumerate(dl):
        print("hey", sample.timestamps, [x.shape for x in sample.inputs[0][1]])
        continue
        import pdb; pdb.set_trace()
        print('yo', i, [len(s) for s in sample], get_date_str(sample[0][-1][0].item()))


if __name__ == '__main__':
    from utils import get_dates, D, levels_medium
    import meshes
    from train_fn import save_img_with_metadata
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2022, 7, 31))])

    load_locations = ['/jersey/','/fast/proc/']
    extra_in_out = ['15_msnswrf', '45_tcc', '034_sstk', '142_lsp', '143_cp', '201_mx2t', '202_mn2t']
    extra_out_only = ['142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']
    extra_all = extra_in_out + extra_out_only
    imesh = meshes.LatLonGrid(load_locations=load_locations,source='era5-28', extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    omesh = meshes.LatLonGrid(load_locations=load_locations,source='era5-28', extra_sfc_vars=extra_all, input_levels=levels_medium, levels=levels_medium)
    
    timesteps = [0]
    data = WeatherDataset(DataConfig(
        inputs=[imesh], outputs=[omesh],
        timesteps=timesteps,
        requested_dates = tdates,
        random_timestep_subset = 2
        ))

    data.check_for_dates()
    #self.data_loader = DataLoader(self.data, batch_size=self.conf.batch_size, shuffle=True, num_workers=2, prefetch_factor=2,worker_init_fn=init_fn)
    dl = torch.utils.data.DataLoader(data, batch_size=1, num_workers=0,shuffle=True)

    os.makedirs('imgs/',exist_ok=True)
    for i, sample in enumerate(dl):
        #for i in range(7):
        #    for j in range(len(sample)):
        #        save_img_with_metadata(f'imgs/img_{i}_{j}.png',sample[j][0].detach().cpu().numpy()[0,:,:,-i])
        print(f'{i}/{len(data)}',[len(s) for s in sample], get_date_str(sample[0][-1][0].item()))
