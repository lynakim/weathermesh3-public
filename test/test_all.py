from utils import *
# from train import *
from unittest.mock import patch
import pytest
import matplotlib.pyplot as plt

#run with `pytest -v test/test_all.py``

def test_time():
    # very simple test just to confrim that all of our date utilities are working and are not annoying 
    # about timezone.
    for d in [datetime(2021,1,1,0,0),datetime(1999,1,1,0,0),datetime(2021,1,1,2,0,tzinfo=timezone.utc)]:
        assert to_unix(d) == to_unix(get_date(to_unix(d))) 
        assert to_unix(d) == d.replace(tzinfo=timezone.utc).timestamp()
        assert to_unix(d) == datetime.fromtimestamp(d.replace(tzinfo=timezone.utc).timestamp()).timestamp()

def verify_array(testing, actual):
    assert testing.shape == actual.shape, "Shapes don't match"
    assert np.allclose(testing, actual,rtol=1e-2), "Values don't match"

def test_run_rt():
    import realtime.run_rt_det as rd
    rd.OUTPUT_PATH = '/tmp'
    # Assumess access to /huge for below test_data, and to /fast/proc for gfs and hres inputs
    TEST_DATA_PATH = "/huge/deep/test_data"
    rd.S3_UPLOAD = False
    rd.TARGET_FORECAST_HOUR = 48
    fz = "2024080700"
    rd.run_if_needed(idempotent=False, forecast_zero=fz, rollout_schedule=None, min_dt=6) #todo: only make this run out to like 48hrs
    
    testing = np.load(rd.OUTPUT_PATH+f'/{fz}+0.MA1l.npy', mmap_mode='r')
    actual = np.load(f'{TEST_DATA_PATH}/{fz}+0.MA1l.npy', mmap_mode='r')
    verify_array(testing, actual)

    testing = np.load(rd.OUTPUT_PATH+f'/{fz}+42.MA1l.npy', mmap_mode='r')
    actual = np.load(f'{TEST_DATA_PATH}/{fz}+42.MA1l.npy', mmap_mode='r')
    verify_array(testing, actual)

    # Cleanup tmp files
    for hr in range(49):
        fn = f'{rd.OUTPUT_PATH}/{fz}+{str(hr)}.MA1l.npy'
        if os.path.exists(fn):
            os.remove(fn)
    print("cleaned up tmp files")

#test_run_rt()
