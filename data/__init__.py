import os 
import torchvision 
import torchvision.transforms as transforms

def generate_mnist():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = '/'.join(dir_path.rstrip('/').split('/')[:-1])

    transform = transforms.Compose(
        [transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=os.path.join(dir_path, 'untracked/data'), train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=os.path.join(dir_path, 'untracked/data'), train=False,
                                        download=True, transform=transform)
    classes = [i for i in range(10)]
    
    return trainset, testset, classes