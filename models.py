import torch.nn as nn
import torch.nn.functional as f


class Model(nn.Module):
    def __init__(self, in_feat=4, h1=12, h2=12, h3=7, out_feat=3):
        super().__init__()
        self.layr1 = nn.Linear(in_feat, h1)
        self.layr2 = nn.Linear(h1, h2)
        self.layr3 = nn.Linear(h2, h3)
        self.outlayr = nn.Linear(h3, out_feat)

    def forward(self, inpts):
        out1 = f.relu(self.layr1(inpts))
        out2 = f.relu(self.layr2(out1))
        out3 = f.relu(self.layr3(out2))
        return self.outlayr(out3)
