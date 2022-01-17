#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import torch library to build neural network
import torch  # Elementory function of tensor is define in torch package
import torch.nn as nn # Several layer architectur is define here
import torch.nn.functional as F # loss function and activation function


# In[ ]:


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


# In[ ]:


import os
import torch
def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[ ]:


######################## Input ###############################
def setFolderLocation(datasets, selectedDataset='', train='train', test='test'):
    global DatasetLoc 
    global SelDataSet 
    global trainDir 
    global testDir 
    global data_dir 
    global zipFile
    
    DatasetLoc = datasets    #'/home/pragnesh/Dataset/'
    SelDataSet = selectedDataset
    trainDir = train
    testDir = test
    data_dir = DatasetLoc+SelDataSet
    # zipFile = False


# In[ ]:


#setFolderLocation('/home/pragnesh/Dataset/', 'IntelIC', 'train', 'test')


# In[ ]:


#print(data_dir)


# In[ ]:


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
def extractData(destLoc):
  fullpath = data_dir+'.zip'
  zip_ref = zipfile.ZipFile(fullpath, 'r') #Opens the zip file in read mode
  zip_ref.extractall(destLoc) #Extracts the files into the /tmp folder
  data_dir = destLoc+'/IntelIC'
  testDir ='val'
  zip_ref.close()


# In[1]:


"""
Choose an apropriate batch size that can be loaded in the current 
enviroment without crashing and also do not choose too big batch even 
if dataset is small because it leads to very few updates per epoch
"""
#################### Create Batch Of Dataset and do data augmentation ###########
bs = 16
ImageSize = 224
def setBatchSize(batchSize=32):
    bs = batchSize
    
def setImageSize(ImageSizeLocal=224):
    global ImageSize
    ImageSize = ImageSizeLocal

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
        trainDir: transforms.Compose([
            transforms.RandomResizedCrop(ImageSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        testDir: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in [trainDir, testDir]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs,
                                                 shuffle=True, num_workers=1)
                  for x in [trainDir, testDir]}

    dataset_sizes = {x: len(image_datasets[x]) for x in [trainDir, testDir]}
    class_names = image_datasets[trainDir].classes
    return dataloaders


# In[ ]:




