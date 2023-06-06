"""
Non-parametric
Entropy based MNIST Classifier 
Stage1 : saving MNIST Traing data entropy
"""
import torch 
import torchvision #[Role]:???
from torch.utils.data import Dataset, DataLoader
class MNISTWarpper(Dataset):
    def __init__(self, root, train, transform):
        self.data = torchvision.datasets.MNIST(root=root, train=train, transform=transform, download=True)
    
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

import os 
import time 
import pickle 
import random
import numpy as np 
import argparse
import datetime 
from tqdm import tqdm 
from distutils.util import strtobool
from omegaconf import OmegaConf

# ==== 🔖 Argument Setting ====
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='config.yaml')
parser.add_argument("--post-fix", default='')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--no-date", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)  #[Role]:???
args = parser.parse_args()

flags = OmegaConf.load(args.config)
for key in vars(args):
    setattr(flags, key, getattr(args, key))

random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.backends.cudnn.deterministic = True

flags.date = datetime.datetime.now().strftime(format="%Y_%m_%d-%H_%M_%S")
flags.save_dir = f"results/seed-{flags.seed}"
if not flags.no_date: #[Role]:???
    flags.save_dir = flags.save_dir + f"-{flags.date}"
flags.start_time = time.time()

if not os.path.exists(flags.save_dir):
    os.makedirs(flags.save_dir)
OmegaConf.save(flags, f'{flags.save_dir}/config.yaml')


# ==== 🔖 Running the Experiment ====
CLS_ENTROPY = [[] for i in range(10)] # holder for the entropy for class samples
SAMPLE_INDEX = [[] for i in range(10)] #[Role]:???

train_dataset = MNISTWarpper(flags.data_path, train=True, transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=1) #[Role]:???
pbar = tqdm(enumerate(train_loader))
for i,(x,y) in pbar:
    entropy = compute_entropy(x) #[Role]:???
    CLS_ENTROPY[y.item()].append(compute_entropy(x))
    SAMPLE_INDEX[y.item()].append(i)
    duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-flags.start_time))
    pbar.set_description(f"[INFO]🧪{__file__}|🍀{flags.save_dir}|⌛️N:({i:.2E}) P:({i / len(train_dataset)*100:.2f}%) D:({duration})|")

# post process for saving CLS as tensor
MIN_SIZE = min([len(CLS_ENTROPY[i]) for i in range(10)]) #[Role]:???
for i in range(10):
    print(f"[INFO] {i}-th class {len(CLS_ENTROPY[i])} --> {MIN_SIZE}")
    CLS_ENTROPY[i] = torch.tensor(CLS_ENTROPY[i][:MIN_SIZE])
    SAMPLE_INDEX[i] = SAMPLE_INDEX[i][:MIN_SIZE]
CLS_ENTROPY = torch.stack(CLS_ENTROPY)
print(f"[INFO] '{flags.save_dir}/cls_entropy.pkl' tensor size: {CLS_ENTROPY.size()}")

# Savae the result 
with open(f'{flags.save_dir}/cls_entropy.pkl', 'wb') as f:
    print(f"[INFO] saved '{flags.save_dir}/cls_entropy.pkl'")
    pickle.dump(CLS_ENTROPY, f)
    
with open(f'{flags.save_dir}/sample_index.pkl', 'wb') as f:
    print(f"[INFO] saved '{flags.save_dir}/sample_index.pkl'")
    pickle.dump(SAMPLE_INDEX, f)
OmegaConf.save(flags, f'{flags.save_dir}/config.yaml')