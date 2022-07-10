import torch
import torch.nn as nn 
import copy 

def construct_rho(**rule_description):
    if "z_plus" in rule_description and rule_description['z_plus'] == True:
        def _z_plus_fn(w):
            w =  torch.nn.Parameter(w.clamp(min=0.0))
            return nn.Parameter(w)
        return _z_plus_fn

    if "gamma" in rule_description:
        g = rule_description['gamma']
        def _gamma_fn(w):
            w = w + torch.nn.Parameter(w.clamp(min=0.0)) * g
            return nn.Parameter(w)
        return _gamma_fn
    else:
        return lambda w : w

def keep_conservative(b): 
    # set bias to 0
    return nn.Parameter(torch.zeros_like(b))

def construct_incr(**rule_description):
    if "epsilon" in rule_description:
        e = rule_description['epsilon']
        return lambda x : x + e
    else:
        return lambda x : x


def clone_layer(layer):
    cloned_layer = copy.deepcopy(layer)
    if hasattr(layer, "weight"):
        cloned_layer.weight = nn.Parameter(layer.weight)
    if hasattr(layer, "bias"):
        cloned_layer.bias = nn.Parameter(layer.bias)
    return cloned_layer