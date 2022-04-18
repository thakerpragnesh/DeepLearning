#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np                          # Basic Array and Numaric Operation
import os                                   # use to access the files 
import tarfile                              # use to extract dataset from zip files
import sys
import zipfile                              # to extract zip file

import torch                                # Provides basic tensor operation and nn operation
import torchvision                          # Provides facilities to access image dataset

import my_utils.loadDataset as dl           # create dataloader for selected dataset
import my_utils.loadModel as lm             # facilitate loading and manipulating models
import my_utils.trainModel as tm            # Facilitate training of the model
import my_utils.initialize_pruning as ip    # Initialize and provide basic parmeter require for pruning
import my_utils.facilitate_pruning as fp    # Compute Pruning Value and many things


# ### Data Loader
# #### We set dataset location and set traind and test location properly

# In[2]:


# set the locationn of the dataset and trai and test data folder name
dl.setFolderLocation(datasets       ='/home/pragnesh/Dataset/',
                     selectedDataset='IntelIC/',
                     train          ='train',
                     test           ='test')
# set the imge properties
dl.setImageSize(224)
dl.setBatchSize = 2
dataLoaders = dl.dataLoader()


# ### Load Model
# #### Load the daved model from 

# In[3]:


#load the saved model if have any
loadModel = True
if loadModel:
    load_path = "/home/pragnesh/Model/VGG_IntelIC_v2"
    device1 = torch.device('cpu')
    #device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    newModel = torch.load(load_path, map_location=torch.device(device1))
else:
    #if dont have any saved trained model download pretrained model for tranfer learning
    newmodel = lm.load_model(model_name='vgg16',number_of_class=6,pretrainval=False,
                             freeze_feature=False,device_l=device1)

#print(newModel)


# ## Trainer

# In[4]:


# logFile = '/home/pragnesh/Dataset/Intel_Image_Classifacation_v2/Logs/ConvModelv2.log'
# #tm.device = torch.device('cpu')
# tm.fit_one_cycle(#set locations of the dataset, train and test data
#                  dataloaders=dataLoaders,trainDir=dl.train_directory,testDir=dl.test_directory,
#                  # Selecat a variant of VGGNet
#                  ModelName='vgg16',model=newModel,device_l=device1,
#                  # Set all the Hyper-Parameter for training
#                  epochs=1, max_lr=0.01, weight_decay=0, L1=0, grad_clip=.1, logFile=logFile)

# #Save the  trained model 
# SavePath = '/home/pragnesh/Model/vgg16-v2'
# torch.save(newModel, SavePath)


# ## Pruning

# #### Pruning Initialization
# 
#         1.  Initialization: blockList,featureList,convidx,prune_count,module
#         2.  ComputeCandidateLayer
#         3.  ComputenewList
#         4.  Call CustomPruning
#         5.  Commit Pruning
#         6.  Update feature list
#         7.  Create new temp model with updated feature list
#         8.  Perform deep copy
#         9.  Train pruned model
#         10. Evalute the pruned model 
#         11. Continue another iteration if required and accepted

# In[5]:


blockList  = []              #ip.getBlockList('vgg16')
featureList= []
convIdx    = []
module     = []
prune_count= []

newList     = []
layer_number=0
st=0
en=0
candidateConvLayer =[]
    
def initializePruning():
    global blockList                 #ip.getBlockList('vgg16')
    global featureList 
    global convIdx     
    global module     
    global prune_count
    
    blockList   = ip.createBlockList(newModel)              #ip.getBlockList('vgg16')
    featureList = ip.createFeatureList(newModel)
    convIdx     = ip.findConvIndex(newModel)
    module      = ip.getPruneModule(newModel)
    prune_count = ip.getPruneCount(module=module,blocks=blockList,maxpr=.1)
    
    global newList
    global layer_number
    global st
    global en
    global candidateConvLayer
    
    newList = []
    layer_number = 0
    st = 0
    en = 0
    candidateConvLayer = []
    
    print(f"Block List   = {blockList}\n"
          f"Feature List = {featureList}\n" 
          f"Conv Index   = {convIdx}\n"
          f"Prune Count  = {prune_count}\n"
          f"Start Index  = {st}\n"
          f"End Index    = {en}\n"
          f"Initial Layer Number = {layer_number}\n"
          f"Empy candidate layer list = {candidateConvLayer}"

         )
initializePruning()
# for i in range(len(module)):
#     print(f"{i+1} :{module[i]}")


# #### Implementing custom pruning process

# In[6]:


def computeCandidateConvLayer(module,blockList,st=0,en=0):
    print("Executing Compute Candidate Convolution Layer")
    global layer_number
    candidateConvLayer = []
    
    for bl in range(len(blockList)):    

        
        if bl==0:
            st = 0
        else:
            st=en
        #endif
        
        en = en+blockList[bl]

        print('\nblock =',bl,'blockSize=',blockList[bl],'start=',st,'End=',en)
        
#         if bl>=2:
#             continue
        newList = []
        candidList = []
        for i in range(st,en):
            layer_number =i
            candidList.append(fp.compute_distance_score(module[i]._parameters['weight'],threshold=2))
        #end_for
        candidateConvLayer.append(candidList)
    return candidateConvLayer
 


# In[7]:


#candidateConvLayer = []
def computeNewList(candidateConvLayer,k):
    print("Executing Compute New List")
    newList = []
    for i in range(len(candidateConvLayer)):
        tuppleList = []
        for j in range(k):
            tuppleList.append(candidateConvLayer[i][j])
        newList.append(tuppleList)
    return newList
    
#newList = computeNewList


# In[8]:


import torch.nn.utils.prune as prune
class KernalPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    def compute_mask(self, t, default_mask):
        print("Executing Compute Mask")
        mask = default_mask.clone()
        #mask.view(-1)[::2] = 0
        size = t.shape
        print(size)
        print(f'Layer Number:{layer_number} \nstart={st} \nlength of new list={len(newList)}')
        for k1 in range(len(newList[layer_number-st])):
            for k2 in range(len(newList[layer_number-st][k1])):
                i= newList[layer_number-st][k1][k2][1]
                j= newList[layer_number-st][k1][k2][0]
                
                #print(f"i= {i} , j= {j}")
                
                mask[i][j] = 0
        return mask
def kernal_unstructured(module, name):
    KernalPruningMethod.apply(module, name)
    return module


# #### After pruning create new model with updated pruning list

# In[9]:


def updateFeatureList(featureList,prune_count,start=0,end=len(prune_count)):
    j=0
    i=start
    while j < end:
        if featureList[i] == 'M':
            i+=1
            continue
        else:
            featureList[i] = featureList[i] - prune_count[j]
            j+=1
            i+=1
    return featureList


# #### Copy the non zero weight value from prune model to new model 

# In[10]:


def deepCopy(destModel,sourceModel):
    print("Deep Copy Started")
    for i in range(len(sourceModel.features)):
        print(".",end="")
        if str(sourceModel.features[i]).find('Conv') != -1:
            size_org = sourceModel.features[i]._parameters['weight'].shape
            size_new = destModel.features[i]._parameters['weight'].shape
            ##print(f"Sise of {i}th layer original model:{size_org}")
            ##print(f"Sise of {i}th layer new model:{size_new}")
            #print(f"feature list[{i}]: {featureList[i]}")
            for fin_org in range(size_org[1]):
                j=0
                fin_new = fin_org
                for fout in range(size_org[0]):
                    if torch.norm(sourceModel.features[i]._parameters['weight'][fout][fin_org]) != 0:
                        fin_new +=1;
                        if j>=size_new[0] or fin_new>=size_new[1]:
                            break
                        
                        t = sourceModel.features[i]._parameters['weight'][fout][fin_org]
                        destModel.features[i]._parameters['weight'][j][fin_new]=t
                        
                        j = j+1


# ### Perform Pruning Blockwise For Each Layer of Block

# In[ ]:


def iterativePruning(newModel,module,blockList,prune_epochs):
    for i in range(prune_epochs):
        # 1.  Initialization: blockList,featureList,convidx,prune_count,module
        
        # 2.  ComputeCandidateLayer
        candidateConvLayer = computeCandidateConvLayer(module=module,blockList=blockList)
        
        # 3.  ComputenewList
        newList = []
        for i in range(len(candidateConvLayer)):
            newList.append( computeNewList(candidateConvLayer[i],1) )
        
        # 4.  Call CustomPruning
        for i in range(len(module)):
            kernal_unstructured(module=module[i],name='weight')
        
        # 5.  Commit Pruning
        for i in range(len(module)):
            prune.remove(module=module[i],name='weight')
        
        # 6.  Update feature list
        featureList = updateFeatureList()
        
        # 7.  Create new temp model with updated feature list
        tempModel = lm.create_vgg_from_feature_list(featureList)
        
        # 8.  Perform deep copy
        deepCopy(tempModel,newModel)
        
        # 9.  Train pruned model
        tm.fit_one_cycle(#set locations of the dataset, train and test data
                         dataloaders=dataLoader,trainDir=dl.trainDir,testDir=dl.testDir,
                         # Selecat a variant of VGGNet
                         ModelName='vgg16',model=tempModel,device_l=device1,
                         # Set all the Hyper-Parameter for training
                         epochs=20, max_lr=0.01, weight_decay=0, L1=0, grad_clip=.1, logFile=logFile)
        
        # 10. Evalute the pruned model 
        
        # 11. Continue another iteration if required and accepted
    
iterativePruning(newModel=newModel,module=module,blockList=blockList,prune_epochs=2)


# In[ ]:





# In[ ]:




