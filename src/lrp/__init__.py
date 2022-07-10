
from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn 
from .modules import *


LookUpTable = {
    "Input" : InputLrp,
    "Linear" : LinearLrp,
    "ReLU" : ReluLrp,
    "Conv2d" : Conv2dLrp,
    "MaxPool2d": MaxPoolLrp,  # treat Max pool as Avg Pooling
    "AvgPool2d" : AvgPoolLrp,
    "AdaptiveAvgPool2d" : AdaptiveAvgPoolLrp,
    "Flatten":FlattenLrp,
    "Dropout" : DropoutLrp,
}

class LRP():
    def __init__(self, layers, rule_descriptions, device,  mean=None, std=None, ):
        super().__init__()
        self.device = device
        self.rule_description = rule_descriptions
        self.original_layers = layers
        self.mean = mean 
        self.std = std
        self.lrp_modules = self.construct_lrp_modules(self.original_layers, rule_descriptions, device)
    
        assert len(layers) == len(rule_descriptions)


    def forward(self, a, y=None, class_specific=True):
        # store activations 
        activations = [torch.ones_like(a)] 
        for i, layer in enumerate(self.original_layers):
            try:
                a = layer(a)
            except Exception as e:
                print("Error:", layer)
                print("Error:", e)
                exit()
            activations.append(a)
        
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]
        # compute LRP 
        prediction_outcome = activations.pop(0)
        score = torch.softmax(prediction_outcome, dim=-1)
        if class_specific:
            if y is None:
                class_index = score.argmax(axis=-1)
            else:
                class_index = y
            class_score = torch.FloatTensor(a.size(0), score.size()[-1]).zero_().to("cuda")
            class_score[:,class_index] = score[:,class_index]
        else:
            class_score = score
        modules = []
        relevances = [ class_score] 
        for (Ai, module) in zip(activations, self.lrp_modules):
            Rj = relevances[-1]
            Ri = module.forward(Rj, Ai)
            relevances.append(Ri)
        output = {
            "R" : relevances[-1],
            "all_relevnaces" : relevances,
            "activations" : activations,
            "prediction_outcome": prediction_outcome
        }
        return output 

    def construct_lrp_modules(self, original_layers, rule_descriptions, device):
        used_names = [] 
        modules = [] 

        for i, layer in enumerate(original_layers):
            rule = rule_descriptions[i]
            for k in rule:
                if k not in ['epsilon', 'gamma', "z_plus"]:
                    raise ValueError(f"Invalid LRP rule {k}")
            if i==0 and self.mean is not None:
                name = "Input"
                lrp_module = LookUpTable["Input"](layer, rule,  self.mean, self.std)
                lrp_module.layer.to(device)
                lrp_module.layer_n.to(device)
                lrp_module.layer_p.to(device)
            else:
                name  = layer.__class__.__name__
                assert name in LookUpTable, f"{name} is not in the LookupTable "
                lrp_module = LookUpTable[name](layer, rule)
                lrp_module.layer.to(device)
            modules.append(lrp_module)
            used_names.append(name)
        
        self.kind_warning(used_names)
        return modules[::-1]

    def kind_warning(self, used_names):
        if "ReLU" not in used_names:
            print(f'[Kind Warning] : ReLU is not in the layers. You should manually add activations.' )
            print(f'[Kind Warning] : Are you sure your model structure excludes ReLU : <{used_names}>?')



import numpy as np 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt 
def process_lrp_before_imshow(R):
    power = 1.0
    b = 10*((np.abs(R)**power).mean()**(1.0/power))

    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    return (R, {"cmap":my_cmap, "vmin":-b, "vmax":b, "interpolation":'nearest'} )


def construct_vgg16_layers_and_rules(model):
    layers = [] 
    rules = [] 
    # Rule is z_plus
    for layer in model.features: # Convolution 
        layers.append(layer)
        rules.append({"z_plus":True, "epsilon":1e-6})
    layers.append(model.avgpool)
    rules.append({"z_plus":True, "epsilon":1e-6})
    layers.append(nn.Flatten(start_dim=1))
    rules.append({"z_plus":True, "epsilon":1e-6})

    # Rule is epsilon 
    for layer in model.classifier: # FCL # 3dense
        layers.append(layer)
        rules.append({"z_plus":False, "epsilon":2.5e-1})
    
    return layers, rules


def construct_lrp(model, device):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    mean = torch.FloatTensor(mean).reshape(1,-1,1,1).to(device)
    std = torch.FloatTensor(std).reshape(1,-1,1,1).to(device)

    model.to(device)
    layers, rules = construct_vgg16_layers_and_rules(model)
    
    lrp_model = LRP(layers, rules, device=device, mean=mean, std=std)
    return lrp_model