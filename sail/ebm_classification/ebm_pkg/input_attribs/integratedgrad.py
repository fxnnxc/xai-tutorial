
import torch 
from torch.autograd import Variable

def make_interpolation(x, M, baseline):
    lst = [] 
    for i in range(M+1):
        alpha = float(i/M)  
        interpolated =x * (alpha) + baseline * (1-alpha)
        lst.append(interpolated.clone())
    return torch.stack(lst)

def ig(model, x, y, **kwrags):
    M = 25
    device = x.device
    baseline = torch.zeros_like(x).to(x.device)
    
    X = make_interpolation(x, M, baseline)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    output = model.forward(X)
    score = torch.softmax(output, dim=-1)
    class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to("cuda")
    class_score[:,y] = score[:,y]
    output.backward(gradient=class_score)

    gradient = X.grad  #A pproximate the integral using the trapezoidal rule
    gradient = (gradient[:-1] + gradient[1:]) / 2.0
    output = (x - baseline) * gradient.mean(axis=0)
    output = output.mean(dim=0) # RGB mean
    output = output.abs()
    return output