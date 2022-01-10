#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[2]:


ImageSize = 64
colorChannel = 3


# In[3]:


def getImageSize():
    return ImageSize
def getColorChannel():
    return colorChannel

def setImageSize(x):
    global ImageSize
    ImageSize = x
def setColorChannel(x):
    global colorChannel
    colorChannel = x    


# In[4]:


datasetPath = ''
def setDatasetPath(path):
    global datasetPath
    datasetPath = path


# In[5]:


class IntelImage():
    def __init__(self,path):
        #global datasetPath  
        self.directory = path
        self.Label = [x[0] for x in os.walk(self.directory)]
        #it add parent directory also in the list while we need only subdirectory
        self.Label.pop(0) #extracting complete path
        self.labelCount = [0]*len(self.Label) # NumberOfClass
        self.Name = os.listdir(self.directory)# Extracting SubDirectory Name
        #if we want to check that number of class and lable of each class uncomment
        #print(labelCount)
        #print(Label)
        self.passed = 0
        self.training_data = []
        
        
    def make_training_data(self):
        print(self.directory)
        for i,label in zip(range(len(self.Label)),self.Label):
            print(f"print the {i}th lable {label}\n")
            for f in os.listdir(label):
                try:
                    path = os.path.join(label,f)
                    if path == label:
                        continue
                    # Read file from selected folder and convert to gray scale
                    img = cv2.imread(path)
                    # Resize image os size 50x50
                    img = cv2.resize(img, (ImageSize, ImageSize))
                    self.training_data.append([np.array(img),i])
                    self.labelCount[i] +=1
                    
                except Exception as e:
                    self.passed +=1
                    print(f"\nPrint Path Which Creates Exception: {path}")
                    print(label)
                    print(str(e))
                    pass
        
        for i in range(len(self.labelCount)):
            print(f"Number of images of {self.Name[i]} is : {self.labelCount[i]} ")
        #check the number of files not process in the given folder 
        print("Number Of passed:", self.passed)
        
        #Randomize dataset so that all class labels are randomly distributed
        np.random.shuffle(self.training_data)
        np.save("/home/pragnesh/Dataset/IntelIC/training1210_data.npy",self.training_data)
        print(f"Number of images{len(self.training_data)}")
