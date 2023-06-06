
import torch 
from torch.autograd import Variable

def make_perturbation(x, M, sigma):
    lst = [] 
    for i in range(M):
        noise = torch.normal(0, sigma, size=x.size()).to(x.device)
        lst.append(x.clone() + noise.clone())
    return torch.stack(lst)

def smoothgrad(model, x, y, **kwrags):
    M = 25
    sigma = 0.15
    device = x.device
    sigma = sigma * (x.max() - x.min())
    X = make_perturbation(x, M, sigma)
    grads = []
    for i in range(M):
        x = Variable(X[i], requires_grad=True).to(device)
        x = x.unsqueeze(0)
        x.retain_grad()
        model.zero_grad()
        output = model(x)
        score = torch.softmax(output, dim=-1)
        class_score = torch.FloatTensor(x.size(0), output.size()[-1]).zero_().to("cuda")
        class_score[:,y] = score[:,y]
        output.backward(gradient=class_score)
        gradient = (x.grad.data).abs()
        grads.append(gradient.squeeze(0))
    del X

    output = torch.mean(torch.stack(grads, dim=0), dim=0)  # perturbation mean
    output = output.mean(dim=0) # RGB mean
    return output

def smoothgrad_energy(model, x,  **kwrags):
    M = 25
    sigma = 0.15
    device = x.device
    sigma = sigma * (x.max() - x.min())
    X = make_perturbation(x, M, sigma)
    grads = []
    for i in range(M):
        x = Variable(X[i], requires_grad=True).to(device)
        x = x.unsqueeze(0)
        x.retain_grad()
        model.zero_grad()
        output = model(x).sum()
        output.backward()
        gradient = (x.grad.data).abs()
        grads.append(gradient.squeeze(0))
    del X

    output = torch.mean(torch.stack(grads, dim=0), dim=0)  # perturbation mean
    output = output.mean(dim=0) # RGB mean
    return output

def smoothgrad_energy_perturbed(model, x,  **kwrags):
    M = 25
    sigma = 0.15
    device = x.device
    sigma = sigma * (x.max() - x.min())
    X = make_perturbation(x, M, sigma)
    grads = []
    for i in range(M):
        x = Variable(X[i], requires_grad=True).to(device)
        x = x.unsqueeze(0)
        x.retain_grad()
        model.zero_grad()
        output = model(x).sum()
        output.backward()
        gradient = (x.grad.data).abs()
        grads.append(gradient.squeeze(0))
    del X

    output = torch.mean(torch.stack(grads, dim=0), dim=0)  # perturbation mean
    output = output.mean(dim=0) # RGB mean
    return output