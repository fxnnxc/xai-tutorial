"""
Non-parametric
Entropy based MNIST Classifier 
Stage1 : saving MNIST Traing data entropy
"""
import torch 
import torchvision 
from torch.utils.data import Dataset, DataLoader
class MNISTWarpper(Dataset):
    def __init__(self, root, train, transform):
        self.data = torchvision.datasets.MNIST(root=root, train=train, transform=transform)
    
    def __getitem__(self, x):
        return self.data[x]
    
    def __len__(self):
        return len(self.data)

def compute_entropy(digit_image):
    assert digit_image.size() == (1, 1, 28,28) 
    digit_image = digit_image.flatten()
    digit_image = digit_image / digit_image.sum() #[Role]:???
    assert abs(digit_image.sum() - 1.0) < 1e-5, digit_image.sum() #[Role]:???
    entropy =  (- digit_image * torch.nan_to_num(digit_image.log())).sum() # \sum - p log p
    assert entropy >=0
    return entropy 

import time 
import pickle 
import random
import numpy as np 
import argparse
from tqdm import tqdm 
from omegaconf import OmegaConf

# ==== 🔖 Argument Setting ====
parser = argparse.ArgumentParser()
parser.add_argument("--exp-path") #[Role]:???
args = parser.parse_args()

flags = OmegaConf.load(f"{args.exp_path}/config.yaml")
for key in vars(args):
    setattr(flags, key, getattr(args, key))

random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.backends.cudnn.deterministic = True

flags.start_time = time.time()
OmegaConf.save(flags, f'{flags.save_dir}/config.yaml')


# ==== 🔖 Running the Experiment ====
CLS_ENTROPY = pickle.load(open(f"{flags.exp_path}/cls_entropy.pkl", mode='rb'))
CLS_MEAN = CLS_ENTROPY.mean(dim=1)



test_dataset = MNISTWarpper(flags.data_path, train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1)
pbar = tqdm(enumerate(test_loader))
Y = []
Y_HAT = [] 

eq = 0  #[Role]:???
for i,(x,y) in pbar:
    entropy = compute_entropy(x)
    y_hat = torch.argmin((CLS_MEAN-entropy).abs()) #[Role]:???
    Y.append(y.squeeze().item()) 
    Y_HAT.append(y_hat.item())
    eq += (y.squeeze()== y_hat).sum() #[Role]:???
    duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-flags.start_time))
    pbar.set_description(f"[INFO]🧪{__file__}|🍀{flags.save_dir}|⌛️N:({i:.2E}) P:({i / len(test_dataset)*100:.2f}%) D:({duration})| accuracy: {eq/(i+1)}:.2f")

# post process for saving Y and Y_HAT as tensor

Y = torch.tensor(Y)
Y_HAT = torch.tensor(Y_HAT)

print(f"[INFO] '{flags.save_dir}/Y.pkl' tensor size: {Y.size()}")
print(f"[INFO] '{flags.save_dir}/Y_HAT.pkl' tensor size: {Y_HAT.size()}")

# Savae the result 
with open(f'{flags.save_dir}/Y_HAT.pkl', 'wb') as f:
    print(f"[INFO] saved '{flags.save_dir}/Y_HAT.pkl'")
    pickle.dump(Y_HAT, f)

with open(f'{flags.save_dir}/Y.pkl', 'wb') as f:
    print(f"[INFO] saved '{flags.save_dir}/Y.pkl'")
    pickle.dump(Y, f)