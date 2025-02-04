from tqdm import tqdm
import time
import pynvml
import matplotlib.pyplot as plt
import importlib
import torch
import model_latlon as mll
import neovis
import os
import pickle
from multiprocessing import Process, shared_memory
import multiprocessing as mp
import numpy as np
import pickle
import time
import PIL
import utils
np.set_printoptions(precision=3)

def stichimg(f1,f2,fout="stitched.png"):
    img1 = PIL.Image.open(f1)
    img2 = PIL.Image.open(f2)

    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)

    new_img = PIL.Image.new('RGB', (new_width, new_height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    new_img.save(fout)

def _anl_attn(attns,mesh):
    ps = []
    for l in range(1):
        for b in range(10):
            b += 1010
            print(f"l:{l} b:{b}")
            p = Process(target=plot_heads,args=(l,b,4,4,attns,mesh))
            #p = Process(target=aaa)
            p.start()
            ps.append(p)
    for p in ps:
        p.join()
    print("Plotting finished")

from itertools import product  
def anl_attn(attns,mesh,wx,multiprocess=False):
    assert 64 in attns[0].shape
    dl = (mesh.lats[0] - mesh.lats[1])*8
    latc,lonc = 29.4, -89.5
    latc,lonc = 41.508922, -138.611393
    ai = [0,2,4,6]; aj = ai
    l = list(range(7))
    lats = latc + dl * np.array([-3,-2,-1,2,3,4])
    lons = lonc + dl * np.array([0,1])
    locs = list(product(lats,lons))
    args_ = list(product(locs,l,ai,aj))
    args_ = [x+(attns,mesh,wx) for x in args_]

    ctx = mp.get_context('fork')
    ps = []
    max_processes = 32
    for args in tqdm(args_):
        while len(ps) >= max_processes:
            for p in ps:
                if not p.is_alive():
                    p.join()
                    ps.remove(p)
            time.sleep(0.1)
        p = ctx.Process(target=plot_heads,args=args)
        p.start()
        ps.append(p)
    for p in ps:
        p.join()            
    print("Plotting finished")

def aaa():
    print("aaa")
    time.sleep(1)

def plot_heads(loc,l,ai,aj,attns,mesh,wx=None,wx_win=12):
    plt.switch_backend('agg')
    at = attns[l]; t_win = np.sqrt(at.shape[2]).astype(int)
    latx,lonx = loc
    Nlat,Nlon,_,_ = wx.shape
    latw = np.argmin(np.abs(mesh.lats - latx)) // t_win
    lonw = np.argmin(np.abs(mesh.lons - lonx)) // t_win
    b = latw * Nlon // t_win + lonw 
    xll,xll_shift = roll_lonlats(mesh,window_size=t_win)
    if l % 2 == 0:
        x = xll
    else:
        x = xll_shift
    lons = x[b,:,:,1][0,:].numpy()
    lats = x[b,:,:,0][:,0].numpy()

    savepath="ignored/vis/im2"
    fbase = f'lat{latw}lon{lonw}l{l}i{ai}j{aj}'
    #print(fbase)
    fname= fbase+'.png'
    os.makedirs(savepath,exist_ok=True)


    if wx is not None:
        plt.clf() ; plt.close()
        fig, axs = plt.subplots(3, 2, figsize=(7, 10))
        uv = wx[:,:,2:4,:]
        z = wx[:,:,0,:]
        q = wx[:,:,5,:]
        t = wx[:,:,1,:]
        wf = (wx_win - t_win) // 2 
        lati = mesh.lat2i(lats); lati = np.arange(max(lati[0]-wf,0),min(lati[-1]+wf+1,Nlat-1)) 
        loni = mesh.lon2i(lons); loni = np.arange(max(loni[0]-wf,0),min(loni[-1]+wf+1,Nlon-1))
        latw = mesh.lats[lati]
        lonw = mesh.lons[loni]
        dd = (lats[0]-lats[1]) / 2
        tlat_min, tlat_max = np.min(lats)-dd, np.max(lats)+dd
        tlon_min, tlon_max = np.min(lons)-dd, np.max(lons)+dd
        wlat_min, wlat_max = np.min(latw)-dd, np.max(latw)+dd
        wlon_min, wlon_max = np.min(lonw)-dd, np.max(lonw)+dd


        gi = np.meshgrid(lati,loni)
        g = np.meshgrid(latw,lonw)
        for i,lev in enumerate([1000,900,700,500,200]):
            row, col = divmod(i, 2); ax = axs[row, col]
            j = utils.levels_small.index(lev)
            ax.imshow(t[gi[0],gi[1],j].T,extent=[wlon_min, wlon_max, wlat_min, wlat_max])
            zs = z[gi[0],gi[1],j] / 9.81
            cp = ax.contour(g[1],g[0],zs, levels=5, colors='k',alpha=0.25)
            ax.clabel(cp, inline=True, fontsize=10)
            uvs = uv[gi[0],gi[1],:,j]
            ax.quiver(g[1],g[0],uvs[:,:,0],uvs[:,:,1],color='red') 
            ax.plot(lons[aj], lats[ai], 'w*')
            ax.set_title(f"Level {lev}")
            #draw a red box around the window

        ax = axs[2, 1]
        ax.imshow(np.sum(q[gi[0],gi[1],:],axis=-1).T,extent=[wlon_min, wlon_max, wlat_min, wlat_max])
        ax.plot(*np.array([[tlon_min, tlon_max, tlon_max, tlon_min, tlon_min], [tlat_min, tlat_min, tlat_max, tlat_max, tlat_min]]), 'r',lw=1)
        ax.plot(lons[aj], lats[ai], 'w*')

        plt.tight_layout()
        wx_path = os.path.join(savepath,f'{fbase}_wx.png')
        plt.savefig(wx_path)

    at = attns[l]
    plt.clf() ; plt.close()
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(f'Attention from swin1.block[{l}]')

    for head in range(16):
        row, col = divmod(head, 4)
        ax = axs[row, col]
        
        o = at[b, head, ai*8+aj, :].reshape(8, 8) * 64
        im = ax.imshow(o)
        
        ax.plot(aj, ai, 'w*')
        ax.set_title(f"Head {head}")
        fig.colorbar(im, ax=ax,shrink=0.7)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=-0.1)
    attn_path = os.path.join(savepath,f'{fbase}_attn.png')
    plt.savefig(attn_path)
    stichimg(wx_path,attn_path,fout=os.path.join(savepath,f'{fbase}_s.png'))
    os.remove(wx_path); os.remove(attn_path)


def roll_lonlats(mesh,window_size=8):
    Nlat,Nlon, = mesh.Lats.shape
    llx = torch.zeros([1,Nlat,Nlon,2])
    llx[:,:,:,0] = torch.from_numpy(mesh.Lats*90)
    llx[:,:,:,1] = torch.from_numpy(mesh.Lons*180)
    x = llx
    shift_size = window_size // 2
    shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    x_windows = mll.window_partition(x, window_size)
    x_windows_shift = mll.window_partition(shifted_x, window_size)
    return x_windows,x_windows_shift
 