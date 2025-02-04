import numpy as np
import matplotlib.pyplot as plt


#th = np.linspace(0, 1, 16)

#D = 8
#w = 1



def token_position_embedding():
    # this is the standard one for when you want to embed position for an arbtrary number of tokens

    num_pos = 1000
    base = 10_000
    embed_dim = 64


    th = np.arange(num_pos)
    m = 1/base**(2*np.arange(embed_dim)/embed_dim)
    ω = np.outer(th, m) * np.pi

    x = np.hstack([np.sin(ω), np.cos(ω)])

    print(x.shape)

    v = x @ x.T

    print(v)

    plt.imsave('ohp.png', v)


def john_emb():
    embed_dim = 16
    n = 32

    # for a give theta value, you compute at period of a si
    # function. 
    # at th=2, the period is 2 
    def get_emb(th):
        e = embed_dim**0.25
        T = 2*e**(th)
        x = 2*np.pi/T * np.arange(embed_dim//2)
        emb = np.concat([np.sin(x),np.cos(x)])
        return emb 
    
    ths = np.linspace(0,1,n)
    x = np.zeros((n**2,2*embed_dim))
    for i in range(n):
        for j in range(n):
            x[i*n+j,:] = np.concat([get_emb(ths[j]),get_emb(ths[i])])
    
    v = x @ x.T

    plt.imsave('ignored/cor.png',v)
    
    plt.subplot(3,1,1)
    plt.plot(v[5])
    plt.subplot(3,1,2)
    plt.plot(v[n//2])
    plt.subplot(3,1,3)
    plt.plot(v[-5])
    plt.savefig('ignored/slice.png')
    pass


def john_emb2():
    import torch
    embed_dim = 16
    n = 500

    # for a give theta value, you compute at period of a si
    # function. 
    # at th=2, the period is 2 
    def get_emb(th):
        B,D = th.shape
        e = embed_dim**0.25
        T = 2*e**(th)
        x = 2*torch.pi/T[:,:,None] * torch.arange(embed_dim//2)[None,None,:]
        emb = torch.concat([torch.sin(x),torch.cos(x)],dim=-1).flatten(start_dim=1)
        return emb 
    
    ths = torch.linspace(0,1,n).reshape(-1,1)
    x = get_emb(torch.concat([ths,ths,ths],dim=1))

    v = x @ x.T

    plt.imsave('ignored/cor.png',v)
    
    plt.subplot(3,1,1)
    plt.plot(v[5])
    plt.subplot(3,1,2)
    plt.plot(v[n//2])
    plt.subplot(3,1,3)
    plt.plot(v[-5])
    plt.savefig('ignored/slice.png')
    pass

john_emb2()
exit()

def thing3():
    import torch
    DEFAULT_POSEMB_NUM = 8
    def sin_posembed(x,num=DEFAULT_POSEMB_NUM):
        B,D = x.shape
        pos = x[:,:,None]*(2**(torch.arange(num,device=x.device,dtype=x.dtype))*torch.pi)[None,None,:].to(x.dtype) # [B,D,1] * [1,1,num] = [B,D,num]
        sin = torch.sin(pos).flatten(start_dim=1)
        cos = torch.cos(pos).flatten(start_dim=1)
        out = torch.concat([x,sin,cos],dim=1)
        return out

    ths = torch.linspace(0,1,500)[:,None]
    x = sin_posembed(ths).numpy()
    print(x.shape)
    v = x @ x.T
    plt.imsave('ohp.png',v)

    l = v[200]
    plt.plot(l)
    #plt.savefig('ohp.png')


def rbf():
    spacial_dim = 2
    emb_dim = 6
    full_emb_dim = emb_dim**2

