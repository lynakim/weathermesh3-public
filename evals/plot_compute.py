import matplotlib.pyplot as plt
import numpy as np
import json
import os


DATAPATH = 'ignored/perf/jsons/'
OUTPATH = 'ignored/perf/'
os.makedirs(OUTPATH,exist_ok=True)

def get_vals(xname,ynames):

    ys = {y:[] for y in ynames}; xs = []
    get_int = lambda x: int(x.split(f'{xname}=')[1].split('.')[0])
    for fn in sorted(filter(lambda x: xname in x, os.listdir(DATAPATH)),key=get_int):
        with open(os.path.join(DATAPATH,fn),'r') as f:
            data = json.load(f)
        x = get_int(fn)
        xs += [x]
        for yname in ynames:
            ys[yname] += [np.median(data[yname])]
    return xs,ys



def plot(var):
    plt.figure(figsize=(5,4))
    plt.grid()
    for val in ['gpu_mem']:
        xs,ys = get_vals(var,[val])
        if 'gpu_mem' in val:
            ys[val] = [y/1024 for y in ys[val]]
        plt.plot(xs,ys[val],'-o',label=val)
        plt.xlabel(var)
        plt.ylabel('memory (GiB)')
    plt.legend()
    plt.savefig(f'{OUTPATH}/{var}-mem.png')

    plt.figure(figsize=(5,4))
    plt.grid()
    for val in ['gpu_time']:
        xs,ys = get_vals(var,[val])
        plt.plot(xs,ys[val],'-o',label=val)
        plt.xlabel(var)
        plt.ylabel('time (s)')
    plt.legend()
    plt.savefig(f'{OUTPATH}/{var}-time.png')

def plot_combined(var):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns
    fig.suptitle('Memory and Time Analysis')

    # Memory plot on the first subplot
    ax1.grid()
    for val in ['gpu_mem']:
        xs, ys = get_vals(var, [val])
        if 'gpu_mem' in val:
            ys[val] = [y / 1024 for y in ys[val]]
        ax1.plot(xs, ys[val], '-o', label=val)
    ax1.set_xlabel(var)
    ax1.set_ylabel('memory (GiB)')
    ax1.legend()

    # Time plot on the second subplot
    ax2.grid()
    for val in ['gpu_time']:
        xs, ys = get_vals(var, [val])
        ax2.plot(xs, ys[val], '-o', label=val)
    ax2.set_xlabel(var)
    ax2.set_ylabel('time (s)')
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout
    plt.savefig(f'{OUTPATH}/{var}-combined.png')
    plt.clf()
    plt.close()



def plot_all():
    things_to_plot = set([x.split("=")[0] for x in os.listdir(DATAPATH) if "=" in x])
    print(things_to_plot)
    for thing in things_to_plot:
        plot_combined(thing)

plot_all()