import torch 
from torch.autograd import Variable

def input_times_gradient(model, x):
    x = x.unsqueeze(0)
    device = x.device
    X = Variable(x, requires_grad=True).to(device)
    X.retain_grad()
    
    output = model.forward(X)
    score = torch.softmax(output, dim=-1)
    class_index = output.argmax(axis=-1)
    class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to("cuda")
    class_score[:,class_index] = score[:,class_index]
    output.backward(gradient=class_score)

    vanilla_gradient = X.grad

    output = {}
    output['grad'] = vanilla_gradient.squeeze(0)
    output['attr'] = (vanilla_gradient.data * X.data).squeeze(0)
    return output

def make_interpolation(x, base, M):
    lst = [] 
    for i in range(M+1):
        alpha = float(i/M)  
        interpolated =x * (alpha) + base * (1-alpha)
        lst.append(interpolated.clone())
    return torch.stack(lst)


def integrated_gradient(model, x, M, y=None, baseline=None, random_steps=0):
    device = x.device
    if random_steps > 0:
        return random_integrated_gradient_several(model, x, M, y, random_steps)
    
    if y is None:
        prediction_class = model(x.unsqueeze(0)).argmax(axis=-1).item()
        y = prediction_class
        
    baseline = baseline.to(device)
    X = make_interpolation(x, baseline, M)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    output = model.forward(X)
    score = torch.softmax(output, dim=-1)
    class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to("cuda")
    class_score[:,y] = score[:,y]
    output.backward(gradient=class_score)

    gradient = X.grad  #A pproximate the integral using the trapezoidal rule
    gradient = (gradient[:-1] + gradient[1:]) / 2.0

    output = {}
    output['attr'] = (x - baseline) * gradient.mean(axis=0)
    return output

def random_integrated_gradient_several(model, x, M, y, random_steps):
    device = x.device
    all_grads = []
        
    if y is None:
        prediction_class = model(x.unsqueeze(0)).argmax(axis=-1).item()
        y = prediction_class
    for i in range(random_steps):
        baseline =torch.rand_like(x)

        X = make_interpolation(x, baseline, M)
        X = Variable(X, requires_grad=True).to(device)
        X.retain_grad()
        
        output = model.forward(X)
        score = torch.softmax(output, dim=-1)
        class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to("cuda")
        class_score[:,y] = score[:,y]

        class_score[:,prediction_class] = score[:,prediction_class]
        output.backward(gradient=class_score)

        gradient = X.grad  #A pproximate the integral using the trapezoidal rule
        gradient = (gradient[:-1] + gradient[1:]) / 2.0
        gradient =  (x - baseline) * gradient.mean(axis=0)
        all_grads.append(gradient)

    output = {}
    output['attr'] = torch.mean(torch.stack(all_grads), axis=0)
    return output


def vanilla_gradient(model, X):
    X = X.unsqueeze(0)
    device = X.device
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    output = model.forward(X)
    score = torch.softmax(output, dim=-1)
    class_index = output.argmax(axis=-1)
    class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to("cuda")
    class_score[:,class_index] = score[:,class_index]
    output.backward(gradient=class_score)

    vanilla_gradient = X.grad

    output = {}
    output['attr'] = vanilla_gradient.squeeze(0)
    return output

def make_perturbation(x, M, sigma=1):
    lst = [] 
    for i in range(M):
        noise = torch.normal(0, sigma, size=x.size()).to(x.device)
        lst.append(x.clone() + noise.clone())
    return torch.stack(lst)


def smooth_gradient(model, x, M, sigma, y=None):
    
    device = x.device
    X = make_perturbation(x, M, sigma)
    X = Variable(X, requires_grad=True).to(device)
    X.retain_grad()
    
    output = model.forward(X)
    score = torch.softmax(output, dim=-1)
    class_score = torch.FloatTensor(X.size(0), output.size()[-1]).zero_().to("cuda")
    if y is None:
        class_index = output.argmax(axis=-1)
    else:
        class_index = y
    class_score[:,class_index] = score[:,class_index]
    output.backward(gradient=class_score)

    gradient = X.grad

    output = {}
    output['attr'] = gradient.sum(0)
    return output

class InputGradient:
    def __init__(self, model, method="vanilla", device="cpu"):
        self.methods = {
            "vanilla" : vanilla_gradient,
            "smooth" : smooth_gradient,
            "input_times_grad" : input_times_gradient,
            "ig" : integrated_gradient
        }
        self.method = self.methods[method]
        self.model = model 

    def forward(self, x, **args):
        return self.method(self.model, x, **args)
        
