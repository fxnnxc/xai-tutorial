import torch 
import numpy as np 
from tqdm import tqdm 

class Sampler():
    def __init__(self, img_size, maxlen,):
        self.img_size = img_size
        self.maxlen = maxlen
        self.buffer = torch.zeros(self.maxlen, *self.img_size)
        self.num_neg_samples = 0
    
    def save(self, path):
        torch.save(self.buffer, path)
    
    def load(self, path):
        self.buffer = torch.load(path)
        self.num_neg_samples = len(self.buffer)
    
    def langevian_dynamics(self, model, init_imgs,lagevian_steps,lagevian_step_size,  return_dynamics=False, verbose=False, reversed=False):
        is_training = model.training
        model = model.eval()
        X_k = init_imgs        
        for p in model.parameters():
            p.requires_grad = False
        X_k.requires_grad = True
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        buffer = [X_k.cpu().clone().detach()] 
        buffer_energy = []

        pbar = range(lagevian_steps)
        if verbose:
            pbar = tqdm(pbar)
        for k in pbar:
            model.zero_grad()
            white_noise = torch.randn(X_k.size(), device=X_k.device)
            white_noise.normal_(0, 0.005)
            X_k.data.add_(white_noise)
            X_k.data.clamp_(min=-1.0, max=1.0)
            
            energy = -model(X_k)  # We model -E(x) instead of E(x) for stability 
            energy.sum().backward()
            X_k.grad.data.clamp_(-0.03, 0.03)
            if reversed: # increase the energy if reversed
                X_k.data.add_(lagevian_step_size * X_k.grad.data)
            else:
                X_k.data.add_(-lagevian_step_size * X_k.grad.data)
            X_k.data.clamp_(min=-1.0, max=1.0)
            X_k.grad.detach_()
            X_k.grad.zero_()
            if return_dynamics:
                buffer.append(X_k.cpu().clone().detach())
                buffer_energy.append(energy.cpu().clone().detach())
        model = model.train()
        if return_dynamics:
            buffer = torch.stack(buffer).permute(1,0,2,3,4)
            buffer_energy = torch.stack(buffer_energy).permute(1,0,2)
        
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)
        X_k = X_k.detach()
        energy =energy.detach()
        return X_k, energy, buffer, buffer_energy 
    
    def sample(self, batch_size, binomial=0.05, random=False):
        if random or self.num_neg_samples < batch_size:
            neg_samples = 2*torch.rand(batch_size, *self.img_size) - 1.0
        else:
            n_new = np.random.binomial(batch_size, binomial)
            new_samples = 2*torch.rand(n_new, *self.img_size) - 1.0
            sample_indices = torch.tensor([np.random.randint(min(self.maxlen, self.num_neg_samples)) for i in range(batch_size-n_new)])
            neg_samples = torch.cat([self.buffer[sample_indices].clone(), new_samples], dim=0)
        return neg_samples
    
    def append(self, batch):
        for s in range(batch.size(0)):
            self.buffer[self.num_neg_samples%self.maxlen] = batch[s].clone().detach()
            self.num_neg_samples += 1
            
