#!/usr/bin/env python
# coding: utf-8

# In[1]:
import my_utils.loadModel as lm
import torch

# In[2]:
vgg11 = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
vgg13 = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
vgg16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
vgg19 = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],


# In[3]:
def getBlockList(modelname):
    blocks = []
    if modelname == 'vgg11':
        blocks    = [1, 1, 2, 2, 2]

    if modelname == 'vgg11bn':
        blocks    = [1, 1, 2, 2, 2]        #prunelist = [0, 3, 6,8, 11,13, 16,18]

    if modelname == 'vgg13':
        blocks    = [2, 2, 2, 2, 2]

    if modelname == 'vgg13bn':
        blocks    = [2, 2, 2, 2, 2]

    if modelname == 'vgg16':
        blocks    = [2, 2, 3, 3, 3]

    if modelname == 'vgg16bn':
        blocks    = [2, 2, 3, 3, 3]
    return blocks


# In[4]:
def createBlockList(newModel):
    blockList = []
    count = 0
    for i in range(len(newModel.features)):
        if str(newModel.features[i]).find('Conv') != -1:
            count+=1
        elif str(newModel.features[i]).find('Pool') != -1:
            blockList.append(count)
            count = 0
    return blockList


# ### Indices of conv layer in vgg11/13/16 are store in this list

# In[5]:
def getConvIndex(modelname):
    feature_list = []
    
    if modelname == 'vgg11':
        feature_list = [0, 3, 6,8, 11,13, 16,18]

    if modelname == 'vgg11bn':
        feature_list = [0, 4, 8,11, 15,18, 22,25]

    if modelname == 'vgg13':
        feature_list = [0,2, 5,7, 10,12, 15,17, 20,22]

    if modelname == 'vgg13bn':
        feature_list = [0,3, 7,10, 14,17, 21,24, 28,31]

    if modelname == 'vgg16':
        feature_list = [0,2, 5,7, 10,12,14, 17,19,21, 24,26,28]

    if modelname == 'vgg16bn':
        feature_list = [0,3, 7,10, 14,17,20, 24,27,30, 34,37,40]
    return feature_list


# In[6]:
def findConvIndex(newModel):
    convListIdx = []
    for i in range(len(newModel.features)):
        if str(newModel.features[i]).find('Conv') != -1:
            convListIdx.append(i)
    return convListIdx


# In[7]:
def getFeatureList(modelname):
    if modelname == 'vgg11':
        return vgg11

    if modelname == 'vgg13':
        return vgg13

    if modelname == 'vgg16':
        return vgg16
    


# In[8]:
def createFeatureList(newModel):
    featureList = []
    for i in range(len(newModel.features)):
        if str(newModel.features[i]).find('Conv') != -1:
            size = newModel.features[i]._parameters['weight'].shape
            n = size[0]
            featureList.append(n)
        if str(newModel.features[i]).find('Pool') != -1:
            featureList.append('M')
    #featureList.pop(-1)
    return featureList


# ### Create a list that contain all the conv layer

# In[9]:
def getPruneModule(newModel):
    convList = []
    for i in range(len(newModel.features)):
        if str(newModel.features[i]).find('Conv') != -1:
            convList.append(newModel.features[i])
    return convList


# #### create a pruncount list which prepare a list of number of channel to be prune from each list from max_pruning_ratio

# In[10]:
def getPruneCount(module,blocks,maxpr):
    j=0
    count = 0
    prune_prob = []
    prune_count = []
    for i in range(len(module)):
        if(count<blocks[j]):
            frac = 5-j
        else:
            count=0
            j+=1
            frac = 5-j
        prune_prob.append(maxpr/frac)    
        count+=1
    for i in range(len(module)):
        size = module[i]._parameters['weight'].shape
        c = int(round(size[0]*prune_prob[i]))
        prune_count.append(c)
    return prune_count
