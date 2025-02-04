from datetime import datetime
from plot_tools import *

from lovely_numpy import lo

from meshes import BalloonData, MicrowaveData
from datasets import IgraDataset, MicrowaveDataset


def test_igra():
    d1 = datetime(2019,1,1)
    d2 = datetime(2019,1,30)
    dataset = IgraDataset(BalloonData(da_timestep_offset=0))
    dates = dataset.get_file_times(d1,d2)
    len1 = len(dates)
    assert type(dates) == np.ndarray; assert len(dates.shape) == 1; assert len(dates) > 24*10; assert len(dates) < 24*30
    dates = dataset.get_path_tots(d1,d2)
    len2 = len(dates)
    assert len1*1.7 > len2 and len2 > len1*0.5, f"len1: {len1}, len2: {len2}"
    dataset.load_data(dates[0])
    pass





from data import WeatherDataset, DataConfig
def test_analysis():
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 2, 21))])

    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium)
    timesteps = [6,24]

    from datasets import AnalysisDataset

    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(imesh), 
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        timesteps=timesteps,
        only_at_z=[0,6,12,18],
        requested_dates = tdates
        ))
    
    data.check_for_dates()
    print(len(data))

def test_allsets():
    tdates = get_dates([(D(1979, 1, 23), D(2019, 12, 28)), (D(2021, 2, 1), D(2024, 2, 21))])

    extra_input = ['45_tcc', '168_2d', '246_100u', '247_100v']
    extra_output = ['15_msnswrf', '45_tcc', '168_2d', '246_100u', '247_100v', '142_lsp', '143_cp', '201_mx2t', '202_mn2t', 
             '142_lsp-6h', '143_cp-6h', '201_mx2t-6h', '202_mn2t-6h']

    imesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_input, extra_sfc_pad=len(extra_output)-len(extra_input), levels=levels_medium, hour_offset=6)
    omesh = meshes.LatLonGrid(source='era5-28',extra_sfc_vars=extra_output, levels=levels_medium, hour_offset=6)

    timesteps = [6,24]

    from datasets import AnalysisDataset

    data = WeatherDataset(DataConfig(
        inputs=[
            AnalysisDataset(omesh), 
            IgraDataset(BalloonData(),is_required=True),
            MicrowaveDataset(MicrowaveData('1bamua'),is_required=True)
        ], 
        outputs=[
            AnalysisDataset(omesh),
        ],    
        timesteps=timesteps,
        only_at_z=[0,6,12,18],
        requested_dates = tdates
        ))
    
    data.check_for_dates()
    print(len(data))
    s = data[0]

    data[10]



if __name__ == '__main__':
    #test_analysis()
    test_allsets()

    #test_igra()

