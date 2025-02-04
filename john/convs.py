import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.ones((1,5,4,4))
torch.manual_seed(1)
torch.set_printoptions(precision=3,edgeitems=10,linewidth=500)
#conv = nn.Conv3d(in_channels=5,out_channels=16,kernel_size=(3,3,3),stride=(2,2,2))

conv = nn.ConvTranspose2d(
    out_channels=5,
    in_channels=5,
    kernel_size=(8,8),
    stride=(4,4),
)
y = conv(x)

if 0:
    y[:,:,-2:,-4:-2] += y[:,:,-2:,:2]  
    y[:,:,-2:,2:4] += y[:,:,-2:,-2:]
    y[:,:,:2,-4:-2] += y[:,:,:2,:2]
    y[:,:,:2,2:4] += y[:,:,:2,-2:]
    # this shit sucks there is sometihng with corners and i hate it     



y[:,:,:,2:4] += y[:,:,:,-2:]
y[:,:,:,-4:-2] += y[:,:,:,:2]
y = y[:,:,:,2:-2]

plt.imsave('ohp.png',y[0,0,:,:].detach().numpy())
exit()
yt = torch.roll(y[:,:,:2,:],y.shape[2]//2,dims=2)
yb = torch.roll(y[:,:,-2:,:],y.shape[2]//2,dims=2)
y[:,:,2:4,:] += yt
y[:,:,-4:-2,:] += yb
#y = y[:,:,2:-2,:]

#y = y[:,:,:,2:-2]
y = y[:,:,2:-2,:]
plt.imsave('ohp.png',y[0,0,:,:].detach().numpy())

#y = y[:,:,1:-1,2:-2,2:-2]
print(y.shape)
print(y[0,0,:,:])

pass