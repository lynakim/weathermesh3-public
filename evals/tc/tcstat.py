from evals.tc.tclib import *
from evals.tc.tc import *
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats



def aaa():
    return ([],[])

all_err = defaultdict(aaa)
tots = []

for sname,times in get_big_storms().items():
    #many_fcst(sname,path,times)
    name = sname.lower()
    #if name not in ['hinnamnor']: continue
    if name in ['roslyn']: continue

    tots.append(name)
    outpath = get_outpath(name)
    os.listdir(outpath)
    cmps = set([x[:-8] for x in os.listdir(outpath) if x.endswith('_cmp.pkl')])
    ress = set([x[:-8] for x in os.listdir(outpath) if x.endswith('_res.pkl')])
    all = cmps.intersection(ress)

    err = defaultdict(aaa)
    def get_stuff(dstr):
        with open(f'{outpath}/{dstr}_res.pkl','rb') as f:
            res = pickle.load(f)
        with open(f'{outpath}/{dstr}_cmp.pkl','rb') as f:
            cmp = pickle.load(f)
        wbdist = list(map(haver_dist,res.realpath,res.predpath))
        cmpdist = cmp.dist
        fhr = cmp.forcast_hours
        if len(wbdist) <= 4: return {}
        cmpdist = [cmpdist[i]  if i < len(cmpdist) else np.nan for i in range(len(wbdist))]
        ret = {fhr[i]:(wbdist[i],cmpdist[i]) for i in range(len(wbdist))}
        return ret
    
    l = list(map(get_stuff,all))
    
    def add_errs(accum,new):
        for d in new:
            for fhr,(wbdist,cmpdist) in d.items():
                accum[fhr][0].append(wbdist)
                accum[fhr][1].append(cmpdist)

    add_errs(err,l)
    add_errs(all_err,l)
    print(name)
    def plot_errs(err,name,avg='mean'):
        global tots
        plt.clf()
        plt.figure(figsize=(7,5))
        #err = {k:v for k,v in err.items() if k <= 148}
        for fhr,(wbdist,cmpdist) in err.items():
            pass
            #plt.plot([fhr]*len(wbdist),wbdist,'b*')
            #plt.plot([fhr]*len(cmpdist),cmpdist,'r*')
        
        nums = {fhr:(sum(~np.isnan(wbdist)),sum(~np.isnan(cmpdist))) for fhr,(wbdist,cmpdist) in err.items()}
        #print(nums)



        if avg == 'mean':
            means = {fhr:(np.mean(wbdist),np.mean(np.array(cmpdist)[~np.isnan(cmpdist)])) for fhr,(wbdist,cmpdist) in err.items()}
            #psd = {fhr:(np.mean(wbdist)+np.std(wbdist),np.mean(np.array(cmpdist)[~np.isnan(cmpdist)])+np.std(np.array(cmpdist)[~np.isnan(cmpdist)])) for fhr,(wbdist,cmpdist) in err.items()}
            #msd = {fhr:(np.mean(wbdist)-np.std(wbdist),np.mean(np.array(cmpdist)[~np.isnan(cmpdist)])-np.std(np.array(cmpdist)[~np.isnan(cmpdist)])) for fhr,(wbdist,cmpdist) in err.items()}
            #psd = {fhr:(np.percentile(wbdist,75.),np.percentile(np.array(cmpdist)[~np.isnan(cmpdist)],75.)) for fhr,(wbdist,cmpdist) in err.items()}
            #msd = {fhr:(np.percentile(wbdist,25.),np.percentile(np.array(cmpdist)[~np.isnan(cmpdist)],25.)) for fhr,(wbdist,cmpdist) in err.items()}
        else:
            assert False
            means = {fhr:(np.median(wbdist),np.median(np.array(cmpdist)[~np.isnan(cmpdist)])) for fhr,(wbdist,cmpdist) in err.items()}
        
        #plt.fill_between(list(psd.keys()),[x[0] for x in psd.values()],[x[0] for x in msd.values()],color='b',alpha=0.2,linewidth=0)
        #plt.fill_between(list(psd.keys()),[x[1] for x in psd.values()],[x[1] for x in msd.values()],color='r',alpha=0.2,linewidth=0)
        plt.plot(list(means.keys()),[x[0] for x in means.values()],'b-',label='WindBorne WeatherMesh')
        plt.plot(list(means.keys()),[x[0] for x in means.values()],'b.')

        plt.plot(list(means.keys()),[x[1] for x in means.values()],'r-',label='NWS GFS')
        plt.plot(list(means.keys()),[x[1] for x in means.values()],'r.')
        #print([(1-x[0]/x[1])*100 for x in means.values()])

        if 0:
            for hour, counts in nums.items():
                plt.text(hour, -20, str(counts[1]), va='bottom', ha='center',fontsize=6,c='r')
                plt.text(hour, -20, str(counts[0]), va='top', ha='center',fontsize=6,c='b')  # Adjust -0.05 to control the gap

            plt.ylim(-50,650)
        plt.title(f'{name}, {avg} error | {len(tots)} storms')
        plt.ylabel('Distance (km)')
        plt.xlabel('Forecast Hour')
        plt.xlim(6,150)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        outpath = get_outpath('summary')
        os.makedirs(outpath,exist_ok=True)
        plt.savefig(f'{outpath}/storm_name={name},_neovis.png',dpi=150)
        plt.close()
    
    #plot_errs(err,name)
    plot_errs(all_err,'all')
    gif = Neovis(f"{get_outpath('summary')}")
    gif.make()

        




