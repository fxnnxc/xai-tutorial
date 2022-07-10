import torch 
import torch.nn as nn 

class ReluLrp(nn.Module):
    def __init__(self, layer, rule):
        super().__init__()
        self.layer = layer

    def forward(self, Rj, Ai):
        return Rj


