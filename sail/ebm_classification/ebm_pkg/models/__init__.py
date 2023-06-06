import torch 
import torch.nn as nn 


def get_model(name, **kwargs):
    print(name)
    model = {
        "cnn": CNN(**kwargs),
        "resnet18": CustomResNet18(**kwargs)
    }[name]
    return model

class CNN(nn.Module):
    def __init__(self, in_channels, cnn_dim, out_features, activation, dropout_p=0.0, **kwargs):
        super().__init__()
        if activation == "leaky_relu":
            activation = nn.LeakyReLU
        elif activation == "swish":
            activation = Swish
        elif activation == "relu":
            activation = nn.ReLU
        else:
            raise ValueError() 
        
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels,16,5,2,4),   activation())
        self.cnn2 = nn.Sequential(nn.BatchNorm2d(16), nn.Conv2d(16,32,3,2,1),  activation())
        self.cnn3 = nn.Sequential(nn.BatchNorm2d(32), nn.Conv2d(32,64,3,2,1),  activation())
        self.cnn4 = nn.Sequential(nn.BatchNorm2d(64), nn.Conv2d(64,64,3,2,1),  activation())
        self.flatten = nn.Flatten()
        if dropout_p !=0.0:
            self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(cnn_dim, out_features)
        
        # self._init_params()
    
    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.flatten(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.out(x)
        
        return x
    
    # def _init_params(self):
    #     for name, module in self.named_modules():
    #         if isinstance(module, nn.Conv2d):
    #             nn.init.kaiming_uniform_(module.weight)
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)
    #         elif isinstance(module, nn.Linear):
    #             nn.init.kaiming_uniform_(module.weight)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
from .utils.resnet import ResNet18
class CustomResNet18(ResNet18):
    def __init__(self, in_channels, cnn_dim, out_features, activation, last_avg_kernel_size, **kwargs):
        if activation == "leaky_relu":
            activation = nn.LeakyReLU
        elif activation == "swish":
            activation = Swish
        elif activation == "relu":
            activation = nn.ReLU
        else:
            raise ValueError() 
        
        super().__init__(in_channels, cnn_dim, out_features, activation, last_avg_kernel_size)
        
    def forward(self, x):
        return super().forward(x)



