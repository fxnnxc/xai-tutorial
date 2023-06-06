import torch 
import torch.nn as nn 

class CNN(nn.Module):
    def __init__(self, in_channels, n_classes, flatten_dim):
        super().__init__()
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels,16,5,2,4),   nn.ReLU())
        self.cnn2 = nn.Sequential(nn.BatchNorm2d(16), nn.Conv2d(16,32,3,2,1),  nn.ReLU())
        self.cnn3 = nn.Sequential(nn.BatchNorm2d(32), nn.Conv2d(32,64,3,2,1),  nn.ReLU())
        self.cnn4 = nn.Sequential(nn.BatchNorm2d(64), nn.Conv2d(64,64,3,2,1),  nn.ReLU())
        self.flatten = nn.Flatten()
        self.out = nn.Linear(flatten_dim, n_classes)
    
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.flatten(x)
        x = self.out(x)
        return x

   