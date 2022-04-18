#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import torch library to build neural network
import torch  # Elementory function of tensor is define in torch package
import torch.nn as nn # Several layer architectur is define here
import torch.nn.functional as F # loss function and activation function


# In[2]:


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


import os
import torch
def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# # It takes location of dataset, selected dataset, name of train and test folder

# In[4]:


######################## Input ###############################
def setFolderLocation(datasets, selectedDataset='', train='train', test='test'):
    global dataset_location 
    global selected_dataset 
    global train_directory 
    global test_directory 
    global data_dir 
    global zipFile
    
    dataset_location = datasets    #'/home/pragnesh/Dataset/'
    selected_dataset = selectedDataset
    train_directory = train
    test_directory = test
    data_dir = dataset_location+selected_dataset
    # zipFile = False


# In[5]:


#Data Prepration
"""
Based on the image size of the dataset choose apropriate values of the color channel and Image Size

Here we can define path to a folder where we can keep all the dataset. 
In the following we are using the zip files. Originally dataset should 
be in the following format DataSetName is parent folder and it should 
contain train and test folder. train and test folder should contain 
folder for each category and images of respective category should be in 
the respective category folder
"""
######################### Data Loading #########################################
def extractData(dest_location):
  fullpath = data_dir+'.zip'
  zip_ref = zipfile.ZipFile(fullpath, 'r') #Opens the zip file in read mode
  zip_ref.extractall(dest_location) #Extracts the files into the /tmp folder
  data_dir = dest_location+'/IntelIC'
  test_directory ='val'
  zip_ref.close()


# In[6]:


"""
Choose an apropriate batch size that can be loaded in the current 
enviroment without crashing and also do not choose too big batch even 
if dataset is small because it leads to very few updates per epoch
"""
#################### Create Batch Of Dataset and do data augmentation ###########
batch_size = 16
image_size = 224
def setBatchSize(batch_size_l=32):
    batch_size = batch_size_l
    
def setImageSize(image_sizeLocal=224):
    global image_size
    image_size = image_sizeLocal


# In[7]:


def dataLoader():
    
    """
    Data Augmentaion generally help in reducing overfitting error during 
    trainng process and thus we are performing randon horizontal flip and 
    random crop during training but during validation as no training happens 
    we dont perform data augmentation
    """
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        train_directory: transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        test_directory: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in [train_directory, test_directory]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=1)
                  for x in [train_directory, test_directory]}

    dataset_sizes = {x: len(image_datasets[x]) for x in [train_directory, test_directory]}
    class_names = image_datasets[train_directory].classes
    return dataloaders


# In[8]:


def dataLoaderEval():
    
    """
    Data Augmentaion generally help in reducing overfitting error during 
    trainng process and thus we are performing randon horizontal flip and 
    random crop during training but during validation as no training happens 
    we dont perform data augmentation
    """
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        train_directory: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        test_directory: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in [train_directory, test_directory]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=1)
                  for x in [train_directory, test_directory]}

    dataset_sizes = {x: len(image_datasets[x]) for x in [train_directory, test_directory]}
    class_names = image_datasets[train_directory].classes
    return dataloaders, data_transforms


# In[ ]:




