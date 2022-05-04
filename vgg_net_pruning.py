#!/usr/bin/env python
# coding: utf-8

# In[1]: import different libraries
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


# In[2]: Set the data loader to load the data for selected dataset
dl.setFolderLocation(datasets       ='/home/pragnesh/Dataset/',
                     selectedDataset='IntelIC/',
                     train          ='train',
                     test           ='test')
# set the imge properties
dl.setImageSize(224)
dl.setBatchSize = 16
dataLoaders = dl.dataLoader()


# In[3]:load the saved model if have any and if dont load a standard model
loadModel = True
if loadModel:
    load_path = "/home/pragnesh/Dataset/Intel_Image_Classifacation_v2/Model/VGG_IntelIC_v1-vgg16"
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    newModel = torch.load(load_path, map_location=torch.device(device1))
else:
    #if dont have any saved trained model download pretrained model for tranfer learning
    newmodel = lm.load_model(model_name='vgg16',number_of_class=6,pretrainval=False,
                             freeze_feature=False,device_l=device1)


# In[4]: Chnge the print statement to write and store the intermidiate result in selected file
outLogFile = "/home/pragnesh/Logs/outLogFile.log"

# In[5]: Initialize all the list and parameter
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
    with open(outLogFile, "a") as f:
        
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

        f.write(f"\nBlock List   = {blockList}"
              f"\nFeature List = {featureList}" 
              f"\nConv Index   = {convIdx}"
              f"\nPrune Count  = {prune_count}"
              f"\nStart Index  = {st}"
              f"\nEnd Index    = {en}"
              f"\nInitial Layer Number = {layer_number}"
              f"\nEmpy candidate layer list = {candidateConvLayer}"
             )
        f.close()
initializePruning()


# In[6]: Computer candidate convolution layer  Blockwise
def compCandConvLayerBlkwise(module,blockList,blockId,st=0,en=0,threshold=1):
    with open(outLogFile, "a") as f:
        f.write(f"\nExecuting Compute Candidate Convolution Layer")
    f.close()
    global layer_number
    candidateConvLayer = []
    
    for bl in range(len(blockList)):    
        if bl==0:
            st = 0
        else:
            st=en
        en = en+blockList[bl]
        
        if bl!= blockId:
            continue


        with open(outLogFile, "a") as f:
             print('\nblock =',bl,'blockSize=',blockList[bl],'start=',st,'End=',en)
        f.close()
        newList = []
        candidList = []
        for lno in range(st,en):
            #layer_number =st+i
            with open(outLogFile,'a') as f:
                f.write(f"\nlno in compute candidate {lno}")
            f.close()
            candidateConvLayer.append( fp.compute_distance_score(module[lno]._parameters['weight'],
                                                                n=1, dim_to_keep=[0,1],threshold=1) )
        break
    return candidateConvLayer


# In[7]: Extract k_kernel elements form candidate conv layer and store them in the newlist
def computeNewList(candidateConvLayer,k_kernel):
    with open(outLogFile, "a") as f:
        f.write(f"\nExecuting Compute New List")
    f.close()
    newList = []
    for i in range(len(candidateConvLayer)):#Layer number
        inChannelList = []
        for j in range( len(candidateConvLayer[i]) ) :#Input channel
            tuppleList = []
            for k in range(k_kernel): # extract k kernel working on each input channel
                tuppleList.append(candidateConvLayer[i][j][k])
            inChannelList.append(tuppleList)
        newList.append(inChannelList)
    return newList


# In[8]:Define Custom Pruning
import torch.nn.utils.prune as prune
class KernalPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    def compute_mask(self, t, default_mask):
        with open(outLogFile, "a") as f:
            f.write(f"\nExecuting Compute Mask")
        f.close()
        mask = default_mask.clone()
        #mask.view(-1)[::2] = 0
        size = t.shape
        print(f"\n{size}")
        with open(outLogFile, "a") as f:
            f.write(f'\nLayer Number:{layer_number} \nstart={st} \nlength of new list={len(newList)}')
        f.close()
        for k1 in range(len(newList)):
            for k2 in range(len(newList[layer_number-st][k1])):
                i= newList[layer_number-st][k1][k2][1]
                j= newList[layer_number-st][k1][k2][0]
                if (k1==j):
                    print(":")
                #print(f"i= {i} , j= {j}")
                
                mask[i][j] = 0
        return mask

def kernal_unstructured(module, name):
    KernalPruningMethod.apply(module, name)
    return module


# In[9]: Update the feature list to create new temperoty model work as compress prune model
def updateFeatureList(featureList,prune_count,start=0,end=len(prune_count)):
    with open(outLogFile,"a"):
        f.write("update the feature list\n")
    f.close()
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


# In[10]: copy the non zero elements form pruned model in the compress temp model
def deepCopy(destModel,sourceModel):
    with open(outLogFile, "a") as f:
        f.write("Deep Copy Started")
    f.close()

    for i in range(len(sourceModel.features)):
        print(".",end="")
        if str(sourceModel.features[i]).find('Conv') != -1:
            size_org = sourceModel.features[i]._parameters['weight'].shape
            size_new = destModel.features[i]._parameters['weight'].shape
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


# In[11]: Set the location of the log files to record the results
logFile = '/home/pragnesh/Logs/result.log'
outFile = '/home/pragnesh/Logs/lastResult.log'


from datetime import date

today = date.today()
d1 = today.strftime("%d/%m/%Y")
with open(logFile,'a') as f:
    f.write(f'log for the execution on {d1}')
f.close()

# In[12 ]: Perform iterative pruning
import gc
def iterativePruningBlockwise(newModel,module,blockList,prune_epochs):

    with open(outLogFile,"a") as f:
        f.write("Prunig Process Start\n")
    f.close()
    pc = [1,3,9,26,51]
    for e in range(prune_epochs):
        # 1.  Initialization: blockList,featureList,convidx,prune_count,module
        layerIndex=0
        start = 0
        end = len(blockList)
      
        for blkId in range(start,end):
            # 2 Compute the distance between kernel for candidate convolution layer
            candidateConvLayer = compCandConvLayerBlkwise(module=module,blockList=blockList,blockId=blkId)
            
            # 3 Arrange the element of CandidateConvLaywer in ascending order of their distance
            for j in range(len(candidateConvLayer)):
                fp.sort_kernel_by_distance(candidateConvLayer[j])
            
            # 4 Extract element equal to prune count for that layer
            newList = computeNewList(candidateConvLayer,pc[blkId])
            del(candidateConvLayer[:])
            del(candidateConvLayer)
            
            # 5 perform Custom pruning where we mask the prune weight
            for j in range(blockList[blkId]):
                if blkId<2:
                    layer_number = (blkId*2) + j
                if blkId>=2:
                    layer_number = 4 + (blkId-2)*3+j
                kernal_unstructured(module=module[layer_number],name='weight')
            del(newList)
            gc.collect()
            #layer_number=layerIndex
            #layerIndex +=1
            
                
        # 6.  Commit Pruning
        with open(outLogFile,'a') as f:
            f.write("commit the pruning")
        f.close()


        for i in range(len(module)):
            prune.remove(module=module[i],name='weight')
        
        # 7.  Update feature list
        global featureList
        featureList = updateFeatureList(featureList,prune,start=0,end=len(prune_count))
        
        # 8.  Create new temp model with updated feature list
        tempModel = lm.create_vgg_from_feature_list(featureList)
        
        # 9.  Perform deep copy
        lm.freeze(tempModel,'vgg16')
        deepCopy(tempModel,newModel)
        lm.unfreeze(tempModel)
        
        #10.  Train pruned model
        with open(logFile,'a') as f:
            f.write(f'output of the {e}th iteration is written below\n\n')
        f.close()
        tm.fit_one_cycle(#set locations of the dataset, train and test data
                         dataloaders=dataLoaders,trainDir=dl.trainDir,testDir=dl.testDir,
                         # Selecat a variant of VGGNet
                         ModelName='vgg16',model=tempModel,device_l=device1,
                         # Set all the Hyper-Parameter for training
                         epochs=2, max_lr=0.01, weight_decay=0.01, L1=0.01, grad_clip=.1, logFile=logFile)
        
        # # 10. Evalute the pruned model 
        trainacc = 0
        testacc  = 0
        trainacc = tm.evaluate(newModel,dataLoaders[dl.trainDir])
        testacc  = tm.evaluate(newModel,dataLoaders[dl.testDir])

        with open(outFile,'a') as f:
            f.write(f'output of the {e}th iteration is written below\n\n')
        f.close()
        with open(outFile,'a') as f:
            f.write(f"Train Accuracy :  {trainacc}\n Test Accuracy  :  {testacc} \n")
        f.close()
        savePath = f'/home/pragnesh/Model/vgg16_IntelIc_Prune_{e}'
        torch.save(tempModel,savePath)
    
iterativePruningBlockwise(newModel=newModel,module=module,blockList=blockList,prune_epochs=10)
