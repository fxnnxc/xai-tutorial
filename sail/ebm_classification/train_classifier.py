
from ebm_pkg.datasets import get_datasets
from ebm_pkg.models import get_model
from ebm_pkg.ebm.sampler import Sampler
from ebm_pkg.loss import LossManager
from omegaconf import OmegaConf
import random 
import numpy as np 
import torch 
import argparse
import time 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
from distutils.util import strtobool
import datetime 
date = datetime.datetime.now().strftime(format="%Y-%m-%d--%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--model")
parser.add_argument("--config")
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--ebm-path")
parser.add_argument("--data-path", default='untracked')
parser.add_argument("--loss", type=str)
parser.add_argument("--train-type", type=str, default='uniform_neg')
parser.add_argument("--pseudo-label-steps", default=60, type=int)
parser.add_argument("--pseudo-label-step-size", default=10, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--no-regularization",  type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)

args = parser.parse_args()
flags = OmegaConf.load(args.config)
flags.batch_size = flags.batch_size // 2  # For positive and negative samples 
flags.date = date
for key in vars(args):
    setattr(flags, key, getattr(args, key))

random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.backends.cudnn.deterministic = True

flags.save_path = f"results/train/{flags.data}/{flags.train_type}/{flags.loss}/seed_{flags.seed}"
import os 
if not os.path.exists(flags.save_path):
    os.makedirs(f"{flags.save_path}")
if not os.path.exists(flags.data_path):
    os.makedirs(flags.data_path)

if flags.train_type == "pseudo_label":
    assert flags.loss == "pseudo_label", "only pseudo_label loss can be used for train_type: pseudo_label"

if flags.train_type == "normal":
    assert flags.loss == "normal", "only normal loss can be used for train_type: normal"



# ------Load EBM -----
ebm_flags = OmegaConf.load(f"{flags.ebm_path}/config.yaml")
configs = {
    "cnn" : (1, ebm_flags.activation, ebm_flags.cnn_dim, None), 
    "resnet18" : (1, ebm_flags.activation, ebm_flags.cnn_dim, ebm_flags.avg_pool_size)
}
out_features, activation, cnn_dim, last_avg_kernel_size = configs[ebm_flags.model]
ebm_model = get_model(ebm_flags.model, 
                in_channels=ebm_flags.in_channels,
                out_features=out_features,
                activation=activation,
                cnn_dim=cnn_dim,
                last_avg_kernel_size=last_avg_kernel_size)
ebm_model.load_state_dict( torch.load(f"{flags.ebm_path}/model.pt"))
ebm_model.to(flags.device)
sampler = Sampler(ebm_flags.img_size, ebm_flags.maxlen)
sampler.load(f"{flags.ebm_path}/buffer.pt")
# -------- End Loading EBM / Make Classifier -------------
configs = {
    "cnn" : (flags.num_classes, 'relu' , flags.cnn_dim, None), 
    "resnet18" : (flags.num_classes, 'relu', flags.cnn_dim, flags.avg_pool_size)
}
out_features, activation, cnn_dim, last_avg_kernel_size = configs[flags.model]
model = get_model(flags.model, 
                in_channels=flags.in_channels,
                out_features=out_features,
                activation=activation,
                cnn_dim=cnn_dim,
                dropout_p = flags.dropout_p,
                last_avg_kernel_size=last_avg_kernel_size)

# -------- End  Making Classifier -------------

train_dataset, valid_dataset = get_datasets(flags.data, data_path=flags.data_path)
train_loader =  DataLoader(train_dataset, batch_size=flags.batch_size)
valid_loader = DataLoader(valid_dataset, batch_size=flags.batch_size) 
 
model.to(flags.device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=flags.learning_rate)
lr_scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: flags.learning_rate_decay**epoch)

writer = SummaryWriter(os.path.join(flags.save_path, "runs", flags.date ))
flags.count = 0
flags.current_performance = 0
flags.best_performance = 0

flags.start_time = time.time()
flags.results = []
optimizer.zero_grad()
loss_manager = LossManager()
for epoch in range(flags.epoch_num):
    # training
    model.train()
    flags.epoch = epoch
    pbar = tqdm((train_loader), total=len(train_loader.dataset)//flags.batch_size)
    for batch_idx, (X_pos, Y_pos) in enumerate(pbar):
        # --- [Start] EBM logic for negative samples 
        if flags.train_type =="normal":
            X_pos = X_pos.to(flags.device)
            X_pos.data.clamp_(min=-1.0, max=1.0) 
            Y_pos = Y_pos.to(flags.device)
                    
            # ----- Forward Logic    
            Y_hat_pos = model(X_pos)

            # ----- Loss Computation 
            loss = getattr(loss_manager, flags.loss)(flags, 
                                                    writer,  
                                                    pbar,
                                                    Y_pos,
                                                    Y_hat_pos, 
                                                    None, 
                                                    None, 
                                                    None)
        elif flags.train_type =="robust":
            X_pos = X_pos.to(flags.device)
            Y_pos = Y_pos.to(flags.device)
                    
            # ----- Forward Logic   
            white_noise = torch.randn(X_pos.size(), device=X_pos.device)
            white_noise.normal_(0, 0.005)
            X_pos.data.add_(white_noise)
            X_pos.data.clamp_(min=-1.0, max=1.0) 
            Y_hat_pos = model(X_pos)

            # ----- Loss Computation 
            loss = getattr(loss_manager, flags.loss)(flags, 
                                                    writer,  
                                                    pbar,
                                                    Y_pos,
                                                    Y_hat_pos, 
                                                    None, 
                                                    None, 
                                                    None)        
    
        else:
            dic = {}
            if flags.train_type =="uniform_neg":
                sampled_init_fake = sampler.sample(batch_size=X_pos.size(0), binomial=flags.binomial, random=False).to(flags.device)
                X_neg, energy_neg, buffer, energy_buffer = sampler.langevian_dynamics(ebm_model, sampled_init_fake, flags.lagevian_steps, flags.lagevian_step_size)
                sampler.append(X_neg)
            elif flags.train_type == "pseudo_label":
                steps = np.random.randint(flags.pseudo_label_step_size)
                X_pos_copy = X_pos.clone().to(flags.device)
                X_neg, energy_neg, buffer, energy_buffer = sampler.langevian_dynamics(ebm_model, X_pos_copy, flags.pseudo_label_steps, steps, reversed=True)
                dic['pseudo_label_alpha'] = steps/flags.pseudo_label_step_size
             
            # --- [End] EBM logic for negative samples / [Start] Adding White noise to the positive samples 
            X_pos = X_pos.to(flags.device)
            Y_pos = Y_pos.to(flags.device)
            white_noise = torch.randn(X_pos.size(), device=X_pos.device)
            white_noise.normal_(0, 0.005)
            X_pos.data.add_(white_noise)
            X_pos.data.clamp_(min=-1.0, max=1.0)
            with torch.no_grad():
                energy_pos = -ebm_model(X_pos)
                    
            # ----- Forward Logic
            batch = torch.concat([X_pos, X_neg], dim=0)
            batch_output = model(batch)
            Y_hat_pos, Y_hat_neg = batch_output.chunk(2, dim=0)
            
            # ----- Loss Computation 
            loss = getattr(loss_manager, flags.loss)(flags, 
                                                    writer,  
                                                    pbar,
                                                    Y_pos,
                                                    Y_hat_pos, 
                                                    Y_hat_neg, 
                                                    energy_pos, 
                                                    energy_neg,
                                                    **dic)
        # --- Optimize 
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # stablization
        optimizer.step() 


    # eval
    model.eval()
    eq = 0
    for k, (x, y) in (enumerate(valid_loader)):
        x = x.to(flags.device)
        x.data.clamp_(min=-1.0, max=1.0) 
        y = y.to(flags.device)
        y_hat = model.forward(x).argmax(dim=-1)  # [Role]:???
        eq += (y == y_hat).sum()  
    
    flags.current_performance = (eq/len(valid_dataset)).item()
    writer.add_scalar(f"Valid-ACC", flags.current_performance, epoch)
    
    if flags.current_performance > flags.best_performance:
        flags.best_performance = flags.current_performance
        torch.save(model.state_dict(), os.path.join(flags.save_path, f"model_best.pt"))
        OmegaConf.save(flags, f"{flags.save_path}/config.yaml")
    
    if epoch % flags.save_epochs==0 or epoch == flags.epoch_num -1:
        path = f"{flags.save_path}/checkpoint_{epoch}"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), f"{path}/model.pt")
        
        flags.results.append({'epoch': epoch, 
                                'performance':flags.current_performance, 
                                'last_lr':lr_scheduler.get_last_lr()[0]})
        OmegaConf.save(flags, f"{path}/config.yaml")
        
    lr_scheduler.step() #[Role]:???
    