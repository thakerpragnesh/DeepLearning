#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import torch library to build neural network
import torch  # Elementory function of tensor is define in torch package
import torch.nn as nn # Several layer architectur is define here
import torch.nn.functional as F # loss function and activation function


# In[3]:


"""
Computer vision is one of the most important application and thus lots 
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


def loadModel(modelname,numberOfClass,pretrainval=False,freezeFeature=False,device=torch.device('cpu')):
    #device = device = torch.device('cpu')
    #device = getDeviceType()
    if modelname == 'vgg16' or modelname == 'vgg13' or modelname == 'vgg11':
        if modelname == 'vgg11':
            newModel = torchvision.models.vgg11(pretrained=pretrainval)
        if modelname == 'vgg13':
            newModel = torchvision.models.vgg13(pretrained=False)
        if modelname == 'vgg16':
            newModel = torchvision.models.vgg16(pretrained=False)    
        if freezeFeature:
            for param in newModel.parameters():
                param.requires_grad = False
        #Need to change the below code if we choose different model
        print(newModel.classifier[6])
        num_ftrs = newModel.classifier[6].in_features
        # Here the size of each output sample is set to 10.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        newModel.classifier[6] = nn.Linear(num_ftrs, numberOfClass)
        newModel = newModel.to(device)
        return newModel
    
    if modelname == 'vgg16bn' or modelname == 'vgg13bn' or modelname == 'vgg11bn':
        if modelname == 'vgg11bn':
            newModel = torchvision.models.vgg11_bn(pretrained=pretrainval)
        if modelname == 'vgg13bn':
            newModel = torchvision.models.vgg13_bn(pretrained=False)
        if modelname == 'vgg16bn':
            newModel = torchvision.models.vgg16_bn(pretrained=False)    
        if freezeFeature:
            for param in newModel.parameters():
                param.requires_grad = False
        #Need to change the below code if we choose different model
        print(newModel.classifier[6])
        num_ftrs = newModel.classifier[6].in_features
        # Here the size of each output sample is set to 10.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        newModel.classifier[6] = nn.Linear(num_ftrs, numberOfClass)
        newModel = newModel.to(device)
        return newModel
    
    
    ##################### Download Pretrain ResNet 18 ############################
    if modelname == 'resnet18':
        newModel = torchvision.models.resnet18(pretrained=False)
        print(newModel.fc)
        for param in newModel.parameters():
            param.requires_grad = False
        #print(newModel)
        #Need to change the below code if we choose different model
        num_ftrs = newModel.fc.in_features
        # Here the size of each output sample is set to 10.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        newModel.fc = nn.Linear(num_ftrs, 10)
        newModel = newModel.to(device)
        return newModel


# In[4]:


def loadSavedModel(LoadPath,device):
    if device== torch.device('cpu'):
        newModel = torch.load(LoadPath, map_location=torch.device('cpu'))
    else:
        newModel = torch.load(LoadPath, map_location=torch.device('cuda'))


# In[5]:


def freeze(model,modelName):
    if modelname == 'vgg16':
        for param in model.parameters():
            if count == 30:
              param.requires_grad=True
            else:
              param.requires_grad=False
                
    if modelname == 'vgg13':
        for param in model.parameters():
            if count == 24:
              param.requires_grad=True
            else:
              param.requires_grad=False
            
    if modelname == 'vgg11':
        for param in model.parameters():
            if count == 20:
              param.requires_grad=True
            else:
              param.requires_grad=False


# In[6]:


def freezeFeature(model,modelName):
    if modelname == 'vgg16':
        for param in model.parameters():
            if count in (26,28,30):
              param.requires_grad=True
            else:
              param.requires_grad=False
                
    if modelname == 'vgg13':
        for param in model.parameters():
            if count in (20,22,24):
              param.requires_grad=True
            else:
              param.requires_grad=False
            
    if modelname == 'vgg11':
        for param in model.parameters():
            if count in (16,18,20):
              param.requires_grad=True
            else:
              param.requires_grad=False


# In[7]:


def unfreeze(model,modelName):
    for param in model.parameters():
        param.requires_grad=True

