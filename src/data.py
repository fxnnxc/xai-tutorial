import torch
import torchvision.transforms as transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225] 

mean = torch.FloatTensor(MEAN).reshape(1,-1,1,1).cuda()
std = torch.FloatTensor(STD).reshape(1,-1,1,1).cuda()
preprocess = transforms.Compose([
                            # transforms.ToPILImage(),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=MEAN, std=STD),    
                        ])

preprocess_un = transforms.Compose([
                            # transforms.ToPILImage(),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),   
                        ])
