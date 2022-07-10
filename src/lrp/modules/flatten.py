import torch 
import torch.nn as nn

class FlattenLrp(nn.Module):
    def __init__(self, layer, rule):
        super().__init__()
        self.layer = layer

    def forward(self, Rj, Ai):
        Rj = Rj.view(size=Ai.shape)
        return Rj




