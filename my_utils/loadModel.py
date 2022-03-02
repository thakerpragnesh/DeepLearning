#!/usr/bin/env python
# coding: utf-8

# In[1]:
#import torch library to build neural network
import torch  # Elementory function of tensor is define in torch package
import torch.nn as nn # Several layer architectur is define here
import torch.nn.functional as F # loss function and activation function


#In[2]:
"""Computer vision is one of the most important application and thus lots 
of deplopment in the and torch.vision provides many facilities that can 
be use to imporve model such as data augmentation, reading data batchwise, 
suffling data before each epoch and many more
"""
# import torch library related to image data processing
import torchvision # provides facilities to access image dataset
from torchvision.datasets.utils import download_url 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision import datasets, models, transforms


# In[3]:
vgg11_feature_list = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
vgg13_feature_list = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
vgg16_feature_list = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


# In[4]:
def get_device_type():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# In[5]:
def load_model(model_name, number_of_class, pretrainval=False, freeze_feature=False, device_l=torch.device('cpu')):
    device = device_l
    if model_name == 'vgg16' or model_name == 'vgg13' or model_name == 'vgg11':
        if model_name == 'vgg11':
            new_model = torchvision.models.vgg11(pretrained=pretrainval)
        if model_name == 'vgg13':
            new_model = torchvision.models.vgg13(pretrained=pretrainval)
        if model_name == 'vgg16':
            new_model = torchvision.models.vgg16(pretrained=pretrainval)
            print('VGG16 Loaded')
        if freeze_feature:
            for param in new_model.parameters():
                param.requires_grad = False
        #Need to change the below code if we choose different model
        print(new_model.classifier[6])
        num_ftrs = new_model.classifier[6].in_features
        # Here the size of each output sample is set to 10.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        new_model.classifier[6] = nn.Linear(num_ftrs, number_of_class)
        new_model = new_model.to(device)
        return new_model
    
    if model_name == 'vgg16bn' or model_name == 'vgg13bn' or model_name == 'vgg11bn':
        if model_name == 'vgg11bn':
            new_model = torchvision.models.vgg11_bn(pretrained=pretrainval)
        if model_name == 'vgg13bn':
            new_model = torchvision.models.vgg13_bn(pretrained=pretrainval)
        if model_name == 'vgg16bn':
            new_model = torchvision.models.vgg16_bn(pretrained=pretrainval)    
        if freeze_feature:
            for param in new_model.parameters():
                param.requires_grad = False
        #Need to change the below code if we choose different model
        print(new_model.classifier[6])
        num_ftrs = new_model.classifier[6].in_features
        # Here the size of each output sample is set to 10.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        new_model.classifier[6] = nn.Linear(num_ftrs, number_of_class)
        new_model = new_model.to(device)
        return new_model
    
    
    ##################### Download Pretrain ResNet 18 ############################
    if model_name == 'resnet18':
        new_model = torchvision.models.resnet18(pretrained=False)
        print(new_model.fc)
        for param in new_model.parameters():
            param.requires_grad = False
        #print(new_model)
        #Need to change the below code if we choose different model
        num_ftrs = new_model.fc.in_features
        # Here the size of each output sample is set to 10.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        new_model.fc = nn.Linear(num_ftrs, 10)
        new_model = new_model.to(device)
        return new_model


# In[6]:
def load_saved_model(load_path,device):
    if device== torch.device('cpu'):
        return torch.load(load_path, map_location=torch.device('cpu'))
    else:
        return torch.load(load_path, map_location=torch.device('cuda'))
    r


# In[7]:
def freeze(model,model_name):
    count = 0
    if model_name == 'vgg16':
        for param in model.parameters():
            if count == 30:
              param.requires_grad=True
            else:
              param.requires_grad=False
        count = count+1	  
                
    if model_name == 'vgg13':
        for param in model.parameters():
            if count == 24:
              param.requires_grad=True
            else:
              param.requires_grad=False
            
    if model_name == 'vgg11':
        for param in model.parameters():
            if count == 20:
              param.requires_grad=True
            else:
              param.requires_grad=False


# In[8]:
def freeze_feature(model,model_name):
    if model_name == 'vgg16':
        for param in model.parameters():
            if count in (26,28,30):
              param.requires_grad=True
            else:
              param.requires_grad=False
                
    if model_name == 'vgg13':
        for param in model.parameters():
            if count in (20,22,24):
              param.requires_grad=True
            else:
              param.requires_grad=False
            
    if model_name == 'vgg11':
        for param in model.parameters():
            if count in (16,18,20):
              param.requires_grad=True
            else:
              param.requires_grad=False


# In[9]:
def unfreeze(model,model_name):
    for param in model.parameters():
        param.requires_grad=True


# In[10]:
###Define a VggNet Architecture
from typing import Union, List, Dict, Any, cast
class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 10, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
    #    print(count)
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    #    count+=1
    return nn.Sequential(*layers)


# In[ 11]:
def create_vgg_from_feature_list(VggFeatureList, batch_norm: bool=False, progress: bool=False) -> VGG:
    
    #Calling the constructer here
    feature = make_layers(VggFeatureList, batch_norm=batch_norm)
    model = VGG( feature )
    return model
