import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
import time

class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    if rank == 0:
        time.sleep(3)
    loss_fn(outputs, labels).backward()
    torch.cuda.synchronize()
    print(f"rank {rank}, backward completed")
    print(next(ddp_model.named_parameters()))
    print(rank,list(ddp_model.named_parameters())[0][1].grad[0,0].item())
    optimizer.step()
    print(f"rank {rank}, step completed")
    print(rank,list(ddp_model.named_parameters())[0][1].grad[0,0].item())
    dist.destroy_process_group()

if __name__ == "__main__":
    demo_basic()