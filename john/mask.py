import torch


mask = torch.ones(2, 100, 100).tril()
print(mask.shape)
print(mask)