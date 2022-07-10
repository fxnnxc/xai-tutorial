import torch 
import torch.nn as nn 
from .utils import construct_incr, construct_rho, clone_layer, keep_conservative

class LinearLrp(nn.Module):
    def __init__(self, layer, rule):
        super().__init__()

        self.layer = clone_layer(layer)
        self.rho = construct_rho(**rule)
        self.incr = construct_incr(**rule)

        self.layer.weight = self.rho(self.layer.weight)
        self.layer.bias = keep_conservative(self.layer.bias)

    def forward(self, Rj, Ai):
        
        # Ai = torch.autograd.Variable(Ai, requires_grad=True)
        Z = self.layer.forward(Ai)
        Z = self.incr(Z)
        S = (Rj / Z).data 
        (Z * S).sum().backward()
        Ci = Ai.grad 

        return  (Ai * Ci).data


