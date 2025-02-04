import torch
x = torch.rand((1000,1000))
x.to("cuda:0")