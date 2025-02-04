import torch
import torch.nn as nn

class TransmissionTransformer(nn.Module):
    def __init__(self,input_dim=4096,tr_dim=256,output_dim=1,depth=8,num_tokens=16,head_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.tr_dim = tr_dim
        self.output_dim = output_dim
        self.depth = depth
        self.num_tokens = num_tokens
        self.head_dim = head_dim
        self.token_embedding = nn.Linear(input_dim, num_tokens * tr_dim)
        self.tr_layers = nn.TransformerEncoderLayer(d_model=tr_dim, dim_feedforward=tr_dim*4,nhead=tr_dim//head_dim,activation="gelu",batch_first=True)
        self.tr_encoder = nn.TransformerEncoder(self.tr_layers, num_layers=depth)
        self.token_combiner = nn.Linear(num_tokens * tr_dim, output_dim)

    def forward(self,x):
        B,C = x.shape
        assert C == self.input_dim
        x = self.token_embedding(x)
        x = x.reshape(B,self.num_tokens,self.tr_dim)
        x = self.tr_encoder(x)
        x = x.reshape(B,-1)
        x = self.token_combiner(x)
        return x


if __name__ == "__main__":
    device = "cuda"
    x = torch.randn(5,4096).to(device)
    trtr = TransmissionTransformer().to(device)
    y = trtr(x)
    print(y.shape)