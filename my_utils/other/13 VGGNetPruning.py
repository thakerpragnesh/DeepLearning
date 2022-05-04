#!/usr/bin/env python
# coding: utf-8

# 

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


# ## Data Loader

# In[2]:


# set the locationn of the dataset and trai and test data folder name
dl.setFolderLocation(datasets       ='/home/pragnesh/Dataset/',
                     selectedDataset='IntelIC/',
                     train          ='train',
                     test           ='test')
# set the imge properties
dl.setImageSize(224)
dl.setBatchSize = 2
dataLoader = dl.dataLoader()


# ## Load Model

# In[3]:


#load the saved model if have any
load_path = "/home/pragnesh/Model/VGG_IntelIC_v2"
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
newModel = torch.load(load_path, map_location=torch.device('cpu'))

#if dont have any saved trained model download pretrained model for tranfer learning
# newmodel = lm.load_model(model_name='vgg16',number_of_class=6,pretrainval=False,
#                         freeze_feature=False,device_l=device1)

#print(newModel)


# ## Trainer

# In[ ]:


logFile = '/home/pragnesh/Dataset/Intel_Image_Classifacation_v2/Logs/ConvModelv2.log'
#tm.device = torch.device('cpu')
tm.fit_one_cycle(#set locations of the dataset, train and test data
                 dataloaders=dataLoader,trainDir=dl.trainDir,testDir=dl.testDir,
                 # Selecat a variant of VGGNet
                 ModelName='vgg16',model=newModel,device_l=device1,
                 # Set all the Hyper-Parameter for training
                 epochs=1, max_lr=0.01, weight_decay=0, L1=0, grad_clip=.1, logFile=logFile)

#Save the  trained model 
SavePath = '/home/pragnesh/Model/vgg16-v2'
torch.save(newModel, SavePath)


# ## Pruning

# #### Pruning Initialization

# In[4]:


blockList   = ip.createBlockList(newModel)              #ip.getBlockList('vgg16')
featureList = ip.createFeatureList(newModel)
convIdx     = ip.findConvIndex(newModel)
module      = ip.getPruneModule(newModel)
prune_count = ip.getPruneCount(module=module,blocks=blockList,maxpr=.1)
print(f"Block List   = {blockList}\n"
      f"Feature List = {featureList}\n" 
      f"Conv Index   = {convIdx}\n"
      f"Prune Count  = {prune_count}"
      
     )
# for i in range(len(module)):
#     print(f"{i+1} :{module[i]}")


# #### Implementing custom pruning process

# In[5]:


import torch.nn.utils.prune as prune
class KernalPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        #mask.view(-1)[::2] = 0
        size = t.shape
        print(size)
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

# In[6]:


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

# In[7]:


def deepCopy(destModel,sourceModel):
    print("Deep Copy Started")
    for i in range(len(sourceModel.features)):
        print(".")
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


newList = []
layer_number = 0
st = 0
en = 0
candidateConvLayer = []
def pruneWholeModel(module,blockList):
    global st
    global en
    global layer_number
    global candidateConvLayer
    candidateConvLayer = []
    global newList
    newList = []
    
    for bl in range(len(blockList)):    
        
        if bl==0:
            st = 0
        else:
            st=en
        
        en = en+blockList[bl]

        if bl<=3:
            continue
            
        newList = []
        for i in range(st,en):
            layer_number =i
            candidateConvLayer.append(fp.compute_distance_score(module[i]._parameters['weight'],threshold=1))
            #fp.sort_kernel_by_distance(candidateConvLayer[i])
        for i in range(st,en):
            newList.append( fp.get_k_element(channel_list=candidateConvLayer[i],k=prune_count[i]) )
            kernal_unstructured(module=module[i],name='weight')
            prune.remove(module[i], 'weight')
    
pruneWholeModel(module=module,blockList=blockList)
        ###############IncompleteCode############
        
#print("\n\n\nHere is the :",len(candidateConvLayer))


# In[19]:


print(f"Start = {st} and end = {en}")


# 
# ### Create new temp model with updated feature list
# 

# In[ ]:


featureList = updateFeatureList(featureList=featureList,prune_count=prune_count,start=0,end=7)
print(featureList)


# In[ ]:


tempModel = lm.create_vgg_from_feature_list(featureList)
lm.freeze(tempModel,'vgg16')
#print(tempModel)


# In[ ]:


deepCopy(destModel=tempModel,sourceModel=newModel)


# In[ ]:


def iterativePruning(newModel,module,blockList):
    

