from utils import *
from train import *
from unittest.mock import patch
import pytest
import matplotlib.pyplot as plt


@pytest.fixture(scope='module')
def w():
    with patch('sys.argv', ['test_all.py', '-N']):
        args = parse_args()
        w = get_trainer(args)
        yield w 

@pytest.mark.usefixtures('w')
def test_data_time(w):
    idx= np.random.randint(1,len(w.weather_data)-1)
    b1,ts1 = w.weather_data[idx]
    b2,ts2 = w.weather_data[get_date(ts1)]
    #print(b1.shape,b2.shape)
    assert ts1 == ts2
    assert torch.all(b1 == b2)

@pytest.mark.usefixtures('w')
def test_data_delta(w):
    idx= 129
    b1,ts1 = w.weather_data[idx]
    print(ts1)
    print(w.weather_data.timestamps[idx])
    print(get_date(ts1))
    b,ts = w.weather_data[get_date(ts1)]
    bp,tsp = w.weather_data[get_date(ts1+w.conf.DH*60*60)]
    bm,tsm = w.weather_data[get_date(ts1-w.conf.DH*60*60)]
    #print(b.shape)
    assert torch.all(b[1] == bp[0]) , f'idx={idx}'
    assert torch.all(b[1] == bm[2]) , f'idx={idx}' 
    assert torch.all(bm[2] == bp[0]) , f'idx={idx}'
    assert not torch.all(bm[2] == bp[1]) , f'idx={idx}'


@pytest.mark.usefixtures('w')
def test_dataloader_order(w):
    pass
    

@pytest.mark.usefixtures('w')
def test_errors(w):
    return
    with torch.no_grad():
        b,_ = w.weather_data[0]
        b = b.unsqueeze(0)
        print(b.shape)
        r = w.computeErrors(b,b[:,2]-b[:,1],unnorm=False,do_print=False)
        print(r)
        assert r['129_z'][0] == 0

def test_time():
    #assert datetime(2021,1,1,0,0).timestamp() == datetime(2021,1,1,0,0,tzinfo=timezone.utc).timestamp()
    for d in [datetime(2021,1,1,0,0),datetime(1999,1,1,0,0),datetime(2021,1,1,2,0,tzinfo=timezone.utc)]:
        assert to_unix(d) == to_unix(get_date(to_unix(d))) 
        assert to_unix(d) == d.replace(tzinfo=timezone.utc).timestamp()
        assert to_unix(d) == datetime.fromtimestamp(d.replace(tzinfo=timezone.utc).timestamp()).timestamp()

def test_namespace():
    n1 = SimpleNamespace(a=1,b=5)
    n2 = SimpleNamespace(a=1,b=2,c=3)
    update_namespace(n2,n1)
    assert n2.b == 5
    assert n2.c == 3
    n1 = SimpleNamespace(a=1, b=SimpleNamespace(c=2, d=3))
    n2 = SimpleNamespace(a=10, b=SimpleNamespace(c=20, d=30, f=-1), e=40)
    update_namespace(n2,n1)
    assert n2.b.c == 2
    assert n2.b.d == 3
    assert n2.e == 40
    assert n2.b.f == -1


def test_lr():
    w = WeatherTrainer(None,None,None)
    c = w.conf.lr_sched
    c.cosine_period = 100_000
    # f = lambda x: w.computeLR(x,c)
    # lr = np.vectorize(f)(np.arange(100_000))
    # plt.plot(steps)
    
    f = lambda x, r: w.computeLR(x,c,r)
    steps = np.arange(10_000, 50_000)
    lr = np.vectorize(f)(steps, steps - 10_000)
    plt.plot(steps,lr)
    
    plt.grid()
    os.makedirs('ignored/plots',exist_ok=True)
    plt.savefig('ignored/plots/lr.png')

    

def test_drop():
    w = WeatherTrainer(None,None,None)
    c = w.conf.drop_sched
    f = lambda x: w.computeDrop(x,c)
    lr = np.vectorize(f)(np.arange(1_000_000))
    plt.plot(lr)
    plt.grid()
    os.makedirs('ignored/plots',exist_ok=True)
    plt.savefig('ignored/plots/drop.png')
    print('yo')
