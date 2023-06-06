import torch 
from torch.autograd import Variable

def gradient(model, x, y, **kwrags):
    x = Variable(x, requires_grad=True).to(x.device)
    x = x.unsqueeze(0)
    x.retain_grad()
    
    model.zero_grad()
    output = model(x)
    score = torch.softmax(output, dim=-1)
    class_score = torch.FloatTensor(x.size(0), output.size()[-1]).zero_().to("cuda")
    class_score[:,y] = score[:,y]
    output.backward(gradient=class_score)
    return (x.grad.abs()).detach().cpu().squeeze(0).mean(axis=0)

def gradient_energy(model, x, **kwrags):
    x = Variable(x, requires_grad=True).to(x.device)
    x = x.unsqueeze(0)
    x.retain_grad()
    
    model.zero_grad()
    output = model(x)
    output.backward()
    return (x.grad.abs()).detach().cpu().squeeze(0).mean(axis=0)