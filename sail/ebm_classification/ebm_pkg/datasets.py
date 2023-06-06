# ----------- STATIC Variables -----------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CIFAR100_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR100_STD  = [0.2023, 0.1994, 0.2010] 

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

MNIST_MEAN = [0.1307]
MNIST_STD  = [0.3081] 


# ----------- STATIC functions -----------------
import torchvision 
import torchvision.transforms as T
def get_datasets(name, data_path):
    # ---- Define the wrapper if required -----
    mean, std = {
        "cifar10": [CIFAR10_MEAN, CIFAR10_STD],
        "cifar100": [CIFAR100_MEAN, CIFAR100_STD],
        "mnist": [MNIST_MEAN, MNIST_STD],
        "fashion_mnist": [MNIST_MEAN, MNIST_STD], # incorrect!
    }[name]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # ------ CIFAR ---------
    if name =="cifar10":
        train_dataset  = torchvision.datasets.CIFAR10(root=data_path,  
                                                    train=True,  
                                                    download=True,
                                                    transform=transform) 
        valid_dataset  = torchvision.datasets.CIFAR10(root = data_path,  
                                                    train=False,  
                                                    download=True,
                                                    transform =transform) 

    elif name =="cifar100":
        train_dataset  = torchvision.datasets.CIFAR100(root=data_path,  
                                                    train=True,  
                                                    download=True,
                                                    transform=transform) 
        valid_dataset  = torchvision.datasets.CIFAR100(root = data_path,  
                                                    train=False,  
                                                    download=True,
                                                    transform =transform) 
    # ------ ImageNet ---------
    elif name =="imagenet1k":
        # train_dataset = torchvision.datasets.ImageNet(root=data_path, split="train", transform=transform)
        train_dataset = None  # need to be prepared in the future
        valid_dataset = torchvision.datasets.ImageNet(root=data_path, split="val", transform=transform)
    
    elif name == "mnist":
        train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
        valid_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=transform, download=True) 

    elif name == "fashion_mnist":
        train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=transform, download=True)
        valid_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=transform, download=True) 

    else:
        raise ValueError(f"{name} is not implemented data")
    
    return train_dataset, valid_dataset