# From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html

import random 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
import numpy as np 
import time 
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import os 


class Trainer():
    def __init__(self,  flags,
                        model, 
                        sampler, 
                        train_loader, 
                        save_path,
                        alpha,
                        epochs,
                        batch_size,
                        learning_rate):
        
        self.flags = flags
        self.writer = SummaryWriter(os.path.join(save_path, "runs"))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0, 0.999))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.flags.learning_rate_decay**epoch)
        self.batch_size = batch_size
        self.save_path = save_path
        self.pos_lodaer = train_loader 
        self.neg_loader = sampler 
        self.batch_count = 0
        self.epoch = 1
        self.epochs = epochs
        self.alpha = alpha
        self.DEVICE = flags.device
        self.start_time = time.time()
        self.model = model.to(self.DEVICE)
        
        self.running_loss = 0 
        self.running_cd = 0 
        self.running_reg = 0 
        self.running_pos = 0 
        self.running_neg = 0
        
    
    def train_batch(self, index, X_pos, X_neg):
        # X_pos : reduce the energy 
        # X_neg : increase the energy 
        batch = torch.concat([X_pos, X_neg], dim=0)
        batch_output = self.model(batch)
        
        e_pos, e_neg = batch_output.chunk(2, dim=0)
        e_pos_mean = e_pos.mean()
        e_neg_mean = e_neg.mean()
        

        if self.flags.objective == "cd":    
            contrastive_divergence =  - e_pos_mean +  e_neg_mean  # We model f(x) = -E, instead of f(x) = E
        elif self.flags.objective =='softplus':
            contrastive_divergence= nn.functional.softplus(-e_pos_mean + e_neg_mean )
        regularization_loss =  self.alpha * (e_pos**2 + e_neg**2).mean()
        loss = contrastive_divergence +  regularization_loss 
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step()
        self.callback_batch(index, e_pos_mean, e_neg_mean, contrastive_divergence, regularization_loss, loss)
        
    
    def train_epoch(self):
        self.pbar = tqdm(enumerate(self.pos_lodaer), total=len(self.pos_lodaer.dataset)//self.pos_lodaer.batch_size)
        for i, (X_pos, Y_pos) in self.pbar:
            # Sample initial fake data 
            X_pos = X_pos.to(self.DEVICE)
            Y_pos = Y_pos.to(self.DEVICE)
            white_noise = torch.randn(X_pos.size(), device=X_pos.device)
            white_noise.normal_(0, 0.005)
            
            X_pos.data.add_(white_noise)
            X_pos.data.clamp_(min=-1.0, max=1.0)
            
            sampled_init_fake = self.neg_loader.sample(batch_size=self.pos_lodaer.batch_size, random=False).to(self.DEVICE)
            
            # Lagevian Dynamics for negative samples 
            X_neg, energy, buffer, energy_buffer  = self.neg_loader.langevian_dynamics(self.model, sampled_init_fake, self.flags.lagevian_steps, self.flags.lagevian_step_size)
            self.neg_loader.append(X_neg)
            self.train_batch(i, X_pos, X_neg)  # train with X_pos and X_neg

        self.callback_epoch()
    
    
    def callback_epoch(self):
        self.running_loss = 0 
        self.running_cd = 0
        self.running_reg = 0
        self.running_pos = 0 
        self.running_neg = 0
        
        self.flags.epoch = self.epoch 
        self.flags.last_lr = self.lr_scheduler.get_last_lr()[0]
        self.lr_scheduler.step()
        self.writer.add_scalar(f"Train/LearningRate", self.flags.last_lr, self.batch_count )
        torch.save(self.model.state_dict(), f"{self.save_path}/model.pt")
        self.neg_loader.save(f"{self.save_path}/buffer.pt")
        OmegaConf.save(self.flags,f"{self.save_path}/config.yaml")
        
        self.epoch += 1

        
    def callback_batch(self, index, e_pos_mean, e_neg_mean, cd, reg, loss):
        self.batch_count  += 1
        self.running_loss += loss.item()
        self.running_cd   += cd.item()
        self.running_reg  += reg.item()
        self.running_pos  += e_pos_mean.item()
        self.running_neg  += e_neg_mean.item()
        
        self.writer.add_scalar(f"Train/Loss", loss.item(), self.batch_count )
        self.writer.add_scalar(f"Train/ContDiv", cd.item(), self.batch_count )
        self.writer.add_scalar(f"Train/ConvDiv(pos)", e_pos_mean.item(), self.batch_count )
        self.writer.add_scalar(f"Train/ConvDiv(neg)", e_neg_mean.item(), self.batch_count )
        self.writer.add_scalar(f"Train/Regularization", reg.item(), self.batch_count )
        
        
        duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-self.start_time))         
        self.pbar.set_description(f"👉 path:{self.save_path} [E:({self.epoch/self.epochs:.2f}) D:({duration})]" \
                                  +f"🚀  Loss:{self.running_loss/(index+1):.3E} |" \
                                  +f" ContDiv:{self.running_cd/(index+1):.3E}" \
                                  +f"(pos:{self.running_pos/(index+1):.3E}|neg:{self.running_neg/(index+1):.3E}) |" \
                                  +f" Reglrz:{self.running_reg/(index+1):.3E} |")    
        if self.batch_count % self.flags.save_interval ==0:
            path = f"{self.save_path}/checkpoint_{self.batch_count}"
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(self.model.state_dict(), f"{path}/model.pt")
            self.neg_loader.save(f"{path}/buffer.pt")
            OmegaConf.save(self.flags,f"{path}/config.yaml")



if __name__ == "__main__":

    from omegaconf import OmegaConf
    import random 
    import numpy as np 
    import torch 
    import argparse
    import time 
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm 

    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--config")
    parser.add_argument("--activation", default='swish') # swish leaky_relu 
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-path", default='untracked')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save-interval", default=10, type=int)
    args = parser.parse_args()
    flags = OmegaConf.load(args.config)
    for key in vars(args):
        setattr(flags, key, getattr(args, key))
        
    flags.save_path = f"results/ebm/{flags.data}/{flags.model}/seed_{flags.seed}"

    seed=flags.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    import os 
    if not os.path.exists(flags.save_path):
        os.makedirs(f"{flags.save_path}")
    if not os.path.exists(flags.data_path):
        os.makedirs(flags.data_path)
        
    from ebm_pkg.datasets import get_datasets
    from ebm_pkg.models import get_model
    from ebm_pkg.ebm.sampler import Sampler
    train_dataset, valid_dataset = get_datasets(flags.data, data_path=flags.data_path)
    train_loader =  DataLoader(train_dataset, batch_size=flags.batch_size) 
    # valid_loader = DataLoader(valid_dataset, batch_size=flags.batch_size) 
    
    # Model 
    in_channels = flags.in_channels
    configs = {
        "cnn" : (1, flags.activation, flags.cnn_dim, None), 
        "resnet18" : (1, flags.activation, flags.cnn_dim, flags.avg_pool_size)
    }
    out_features, activation, cnn_dim, last_avg_kernel_size = configs[flags.model]
    model = get_model(flags.model, 
                    in_channels=in_channels,
                    out_features=out_features,
                    activation=activation,
                    cnn_dim=cnn_dim,
                    last_avg_kernel_size=last_avg_kernel_size)
    print(model)
    
    model.to(flags.device)
    sampler = Sampler(flags.img_size, flags.maxlen)

    
    trainer = Trainer(flags, model, sampler, train_loader, flags.save_path, flags.alpha, flags.epochs, flags.batch_size, flags.learning_rate)
    for epoch in range(flags.epochs):
        trainer.train_epoch()
    