import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model_latlon.data import get_constant_vars
from utils import * 
from model_latlon.primatives2d import *



def plothb(edges,strides):
    x = np.linspace(0,np.pi/2,1000)
    y = np.cos(x)
    x2 = np.concat(([0],np.array(edges))) * np.pi/2
    y2 = np.ones_like(x2)

    
    x3 = x2
    y3 = np.concat(([1],1/np.array(strides)))

    print(y3) 
    plt.plot(x,y)
    plt.plot(x2,y2)
    plt.step(x3,y3)
    plt.grid()
    plt.savefig('hb/cos.png')

f=1./45
#edges = [26*f,34*f,44*f,1.]
#strides = [1,2,4,12]
edges = [26*f,34*f,1.]
strides = [1,2,4]
if 0:
    edges = [0.5,1]
    strides = [1,2]

EDGES_DEFAULT = edges
STRIDES_DEFAULT = strides
DEFAULT_KERNEL_HEIGHT = 5


class ToHarebrained2d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_height=DEFAULT_KERNEL_HEIGHT,strip_edges=EDGES_DEFAULT,strip_strides=STRIDES_DEFAULT):
        super(ToHarebrained2d, self).__init__()
        assert strip_edges[-1] == 1.0
        assert len(strip_edges) == len(strip_strides)
        self.strides = strip_strides  
        self.kernel_widths = [(kernel_height//2*2)*x+1 for x in self.strides]
        self.conv_center = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_height,self.kernel_widths[0]))
        self.strip_edges = strip_edges
        self.kernel_height = kernel_height  
        self.convs = nn.ModuleList()
        for i in range(1,len(strip_edges)):
            down = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_height,self.kernel_widths[i]),stride=(1,strip_strides[i]))
            self.convs.append(down)

    def forward(self, x):
        B,C,H,W = x.shape   
        for s in self.strides:
            assert W % s == 0, "Needs to be a divisor, also thank the Babylonians"
        assert (H-1) % 2 == 0, f"Height must be odd, got {H}"

        hpad = self.kernel_height//2

        x_pad = earth_pad2d(x,(hpad,0)) #we do pole padding on the input, and then wrap padding is per strip
        #imsave2('hb/downpad.png',x_pad)

        def f2idx(f,dir):
            # f is a float from 0. (center) to 1. (edge)
            # dir specifies which direction from center. -1 is lower index (towards north), 1 is higher index (towards south)
            return H//2 + hpad + int(dir==1) + dir * int(H//2 * f) 
        
        ctp = f2idx(self.strip_edges[0],-1) - hpad
        cbp = f2idx(self.strip_edges[0],1) + hpad
        center = x_pad[:,:,ctp:cbp,:]
        center_pad = earth_pad2d(center, (0,self.kernel_widths[0]//2))
        center_out = self.conv_center(center_pad)

        hemispheres = []
        for dir in [-1,1]: #northside, then southside
            strips = []
            for i in sorted(list(range(1,len(self.strip_edges))),reverse=dir==-1):
                sl = sorted([
                    f2idx(self.strip_edges[i],dir),
                    f2idx(self.strip_edges[i-1],dir)
                    ])
                sl = [sl[0]-hpad,sl[1]+hpad] 
                strip = x_pad[:,:,sl[0]:sl[1],:]
                strip_pad = earth_pad2d(strip,(0,self.kernel_widths[i]//2))
                strip_out = self.convs[i-1](strip_pad)
                strips.append(strip_out)
            hemispheres.append(strips)

        out = hemispheres[0] + [center_out] + hemispheres[1] # north, middle, south

        for t in out:
            #print(t.shape)
            pass

        nB,_,nH,nW = get_full_dims2d(out)
        assert (B,H,W) == (nB,nH,nW), f"Rip, something is bad, {nB},{nH},{nW} vs {B},{H},{W}"

        return out 
    
def get_strides(x,strip_edges=EDGES_DEFAULT,strip_strides=STRIDES_DEFAULT):
    # Returns a vector that length H, giving what the stide down factor is for each lattitude
    B,C,H,W = x.shape
    half = np.zeros(H//2,dtype=np.int32)
    l = 0
    for i in range(len(strip_edges)):
        c = int(strip_edges[i] * H//2)
        half[l:c] = strip_strides[i]
        l = c
    return np.concat((half[::-1],[half[0]],half))

def get_center(strips):
    assert len(strips) % 2 == 1, f"Strips should always be odd length, {len(strips)}"
    return strips[len(strips)//2]

def get_strip(strips,dir,i):
    assert i >= 0 ; assert dir == 1 or dir == -1
    assert len(strips) % 2 == 1, f"Strips should always be odd length, {len(strips)}"
    return strips[len(strips)//2 + dir*i]

def get_full_dims2d(strips):
    H = sum([strip.shape[2] for strip in strips])
    B,C,_,W = get_center(strips).shape
    return B,C,H,W 
    
class FromHarebrained2d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_height=DEFAULT_KERNEL_HEIGHT,strip_edges=EDGES_DEFAULT,strip_strides=STRIDES_DEFAULT):
        super(FromHarebrained2d, self).__init__()
        self.strides = strip_strides  
        self.kernel_widths = [(kernel_height//2*2)*x+1 for x in self.strides]

        # Fun fact: Biases will fuck with wrapping on a conv up. If you want a bais, you need to do that manually and addit it after the wrap.
        # Aren't wrap convolutions fun?
        self.conv_center = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(kernel_height,self.kernel_widths[0]),bias=False) 

        self.strip_edges = strip_edges
        self.kernel_height = kernel_height  
        self.convs = nn.ModuleList()
        for i in range(1,len(strip_edges)):
            up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(kernel_height,self.kernel_widths[i]),stride=(1,strip_strides[i]),bias=False)
            self.convs.append(up)

    def forward(self, strips):
        N = (len(self.strides) - 1) * 2 + 1
        assert len(strips) == N, f"Strip length mismatch: {len(strips)} vs {N}"
        center = get_center(strips)
        B, C, H, W = get_full_dims2d(strips)
        Cout = self.conv_center.out_channels
        hpad = self.kernel_height // 2
        wpadmax = max(self.kernel_widths) // 2
        Hpad = H + 2 * hpad
        Wpad = W + 2 * wpadmax

        # Helpfull context: the Wpadmax above is actuall conservative and much of the time the actuall right hand side
        # will have less padding because the kernel window overlaps the end of the original input.
        # 
        # Consider ConvTranspose with a stride of 2 and a kernel of 7 on an input of 4, shown below
        #
        # WWWCWCWWW | C: Center of Conv, W: Kernel Window
        # PPPIIIIPP | P: Padding that, I: Interior of the image that we want to keep
        # The Padding on the right edge is smaller by one because of the integer dividing of the input size-1 by the stride

        def f2idx(f,dir):
            # f is a float from 0. (center) to 1. (edge)
            # dir specifies which direction from center. -1 is lower index (towards north), 1 is higher index (towards south)
            return H//2 + hpad + int(dir==1) + dir * int(H//2 * f) 

        x_pad = center.new_zeros(B,Cout,Hpad,Wpad)
        wpadc = self.kernel_widths[0] // 2
        c_out = self.conv_center(center)
        ctp = f2idx(self.strip_edges[0],-1) - hpad
        cbp = f2idx(self.strip_edges[0],1) + hpad
        x_pad[:,:,ctp:cbp,wpadmax-wpadc:Wpad-(wpadmax-wpadc)] += c_out

        ostrips = strips[:N//2] + strips[N//2+1:]
        idxs =  list(reversed(range(1,N//2+1))) + list(range(1,N//2+1))
        dirs = [-1]*(N//2) + [1]*(N//2)

        for j,(strip,i,dir) in enumerate(zip(ostrips,idxs,dirs)):
            #print(j)
            sl = sorted([
                f2idx(self.strip_edges[i],dir),
                f2idx(self.strip_edges[i-1],dir)
                ])
            sl = [sl[0]-hpad,sl[1]+hpad] 
            out = self.convs[i-1](strip)
            wpad = self.kernel_widths[i] // 2
            iL = wpadmax - wpad
            iR = iL + out.shape[-1] # saves a lot of work to not have to compute what this is gunna be explicitly, see my long comment at the top 
            x_pad[:,:,sl[0]:sl[1],iL:iR] += out
            
        #imsave2('hb/up1.png',x_pad) 

        x_pad[:,:,:,wpadmax:2*wpadmax] += x_pad[:,:,:,W+wpadmax:] # wrap right side to left
        x_pad[:,:,:,-2*wpadmax:-wpadmax] += x_pad[:,:,:,:-(W+wpadmax)] # wrap left side to right
        
        x = x_pad[:,:,hpad:-hpad,wpadmax:-wpadmax] # and trim
        #imsave2('hb/xpad_end.png',x_pad)
        #imsave2('hb/x_end.png',x)
        return x 
        
class HarebrainedPad2d(nn.Module):
    def __init__(self, channels,kernel_height=DEFAULT_KERNEL_HEIGHT,strip_strides=STRIDES_DEFAULT):
        super(HarebrainedPad2d, self).__init__()
        self.strides = strip_strides
        self.kernel_height = kernel_height
        self.to_outers = nn.ModuleList()
        self.to_inners = nn.ModuleList()
        self.bt_strides  = []
        for i in range(1,len(strip_strides)):
            assert strip_strides[i] % strip_strides[i-1] == 0
            stride = strip_strides[i] // strip_strides[i-1]
            to_outer = nn.Conv2d(channels, channels, kernel_size=(1, 2*stride+1), stride=(1,stride)) 
            to_inner = nn.ConvTranspose2d(channels, channels, kernel_size=(1, 2*stride+1), stride=(1,stride),bias=False)
            self.to_inners.append(to_inner)
            self.to_outers.append(to_outer)


    def forward(self, strips):
        N = (len(self.strides)-1) * 2 + 1
        assert len(strips) == N, f"strip length mismatch, {len(strips)} vs {N}"

        def get_edge_slices(dir,pad):
            if dir == 1: return (slice(None),slice(None),slice(-pad,None),slice(None))   #positive dir gets bottom edge of tensor 
            elif dir == -1: return (slice(None),slice(None),slice(None,pad),slice(None)) #negative dir gets top edge of tensor
            else: assert False

        pad = self.kernel_height // 2
        strips_pad = [earth_pad2d(x,(pad,pad)) for x in strips]

        c_pad = get_center(strips_pad)
        for dir in [-1,1]:
            a = get_strip(strips,dir,1)[get_edge_slices(-dir,pad)]  # invert dir because we need bottom of the top / top of the bottom
            a = self.to_inners[0](a)  
            a = earth_wrap2d(a,get_center(strips).shape[3],self.to_inners[0].kernel_size[1]//2)
            a = earth_pad2d(a, (0,pad))
            c_pad[get_edge_slices(dir,pad)] = a   
        

        def pad_strip(dir,mi):
            #print(f"Conv pads for {mi}, dir={dir}")
            s_pad = get_strip(strips_pad,dir,mi)
            
            # First, pad from the outer
            if mi+1 > N // 2:
                pass # pretty sure we just wann do nothing if pole cause then it's just the default zero pad behavor
            else:
                # o becuase it's the outer tensor
                o = get_strip(strips,dir,mi+1)[get_edge_slices(-dir,pad)] # if we are working on the top side, then the edge we want is the bottom 
                o = self.to_inners[mi](o)
                o = earth_wrap2d(o,get_strip(strips,dir,mi).shape[3],self.to_inners[mi].kernel_size[1]// 2)
                o = earth_pad2d(o, (0,pad))
                s_pad[get_edge_slices(dir,pad)] = o
            
            # now for the pad from in the inner 
            en = get_strip(strips,dir,mi-1)[get_edge_slices(dir,pad)] # if we are working on the top side, then the edge we want it also the top side 
            en = earth_pad2d(en,(0,self.to_outers[mi-1].kernel_size[1] // 2)) # need to pad this before we can conv down
            en = self.to_outers[mi-1](en)
            en = earth_pad2d(en,(0,pad)) # now pad again so that it fits in this tensor for the actual conv
            s_pad[get_edge_slices(-dir,pad)] = en
            
        for i in range(N//2):
            pad_strip(-1,i+1)
            pad_strip(1,i+1)

        return strips_pad
    
class Harify(nn.Module):
    def __init__(self,module):
        super(Harify,self).__init__()
        self.module = module
    
    def forward(self,strips):
        return [self.module(s) for s in strips]
        


class HarebrainedResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.0, group_norms=[16,16],kernel_size=5):
        super(HarebrainedResBlock2d, self).__init__()
        self.kernel_size = kernel_size

        if group_norms[0] is not None: self.norm1 = nn.GroupNorm(group_norms[0], in_channels)
        else: self.norm1 = nn.Identity()
        self.pad1 = HarebrainedPad2d(in_channels,kernel_height=kernel_size)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        if group_norms[1] is not None: self.norm2 = nn.GroupNorm(group_norms[1], out_channels)
        else: self.norm2 = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.pad2 = HarebrainedPad2d(out_channels,kernel_height=kernel_size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=0)

        self.nin_shortcut = (
            nn.Sequential(HarebrainedPad2d(in_channels),Harify(nn.Conv2d(in_channels, out_channels, kernel_size=1)))
            if in_channels != out_channels
            else Harify(nn.Identity())
        )

        if time_emb_dim is not None:
            assert False
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t=None):
        k = self.kernel_size // 2
        h = [self.norm1(xx) for xx in x]
        h = [F.silu(hh) for hh in h]
        h = self.pad1(h)
        h = [self.conv1(hh) for hh in h]

        if t is not None:
            assert False
            time_emb = self.time_mlp(t)
            time_emb = time_emb[:, :, None, None]  # (N, C, 1, 1)
            h = h + time_emb

        h = [self.norm2(hh) for hh in h]
        h = [F.silu(hh) for hh in h]
        h = [self.dropout(hh) for hh in h]
        h = self.pad2(h)
        h = [self.conv2(hh) for hh in h]

        return [hh + xx for hh,xx in zip(h,self.nin_shortcut(x))]



def dev_hb():
    x = torch.ones(1,1,91,180)
    to = weights1(ToHarebrained2d(1,1))
    res1 = weights1(HarebrainedResBlock2d(1,1,group_norms=[None,None]))
    fro = weights1(FromHarebrained2d(1,1))

    x = to(x)
    x = res1(x)
    x = fro(x)    
    #p = find_periodicity(x)
    #print(p)
    #imsave2('hb/res_out.png',x)



if __name__ == "__main__":

    dev_hb()

    plothb(EDGES_DEFAULT,STRIDES_DEFAULT)

