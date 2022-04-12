# In[1]: Importing Libraries
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


# Data Loader
# In[2]:We set dataset location and set traind and test location properly
dl.setFolderLocation(datasets       ='/home/pragnesh/Dataset/',
                     selectedDataset='IntelIC/',
                     train          ='train',
                     test           ='test')
# set the imge properties
dl.setImageSize(224)
dl.setBatchSize = 2
dataLoader = dl.dataLoader()


# Load Model
# In[3]: load the saved model if have any
load_path = "/home/pragnesh/Model/VGG_IntelIC_v2"
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
newModel = torch.load(load_path, map_location=torch.device('cpu'))

# ## Trainer
# In[ ]: Train the model for selected number of epoch
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

# In[4]: Pruning Initialization
blockList   = ip.createBlockList(newModel)              # ip.getBlockList('vgg16')
featureList = ip.createFeatureList(newModel)         # Number of feature map in each convolution layer
convIdx     = ip.findConvIndex(newModel)               # get the index of convolution parameter in newModel.features list
module      = ip.getPruneModule(newModel)           # extract all the conv layer and store them in the module list
prune_count = ip.getPruneCount(module=module,blocks=blockList,maxpr=.1)  # set amount to be pruned from each layer
print(f"Block List   = {blockList}\n"
          f"Feature List = {featureList}\n" 
          f"Conv Index   = {convIdx}\n"
          f"Prune Count  = {prune_count}"
     )

# In[11]:Implementing custom pruning process
newList = []
layer_number = 0
st = 0
en = 0
candidateConvLayer = []

def computeCandidateConvLayer(module,blockList,st=0,en=0):
    global layer_number
    candidateConvLayer = []
    
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
        return candidateConvLayer
            
# In[6]: Extract K element from each module to be pruned and store them in newList
def computeNewList():
    newList = []
    st =0
    en =0
    for bl in range(len(blockList)):    
        if bl==0:
            st = 0
        else:
            st=en
        en = en+blockList[bl]
        print(f"start ={st}, End={en} and last module in the block {module[en-1]._parameters['weight'].shape}")
        newList = []
        print(f"end = {en}")
        for i in range(st,en):
            print(f"i = {i}")
            layer_number=i
            newList.append( fp.get_k_element(channel_list=candidateConvLayer[i-st],k=prune_count[i]) )
    return newList

# In[7]:
import torch.nn.utils.prune as prune
class KernalPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        #mask.view(-1)[::2] = 0
        size = t.shape
        print(size)
        print(f'Layer Number:{layer_number} \nstart={st} \nlength of new list={len(newList)}')
        for k1 in range(len(newList[layer_number-st])):
            for k2 in range(len(newList[layer_number-st][k1])):
                i= newList[layer_number-st][k1][k2][1]
                j= newList[layer_number-st][k1][k2][0]
                mask[i][j] = 0
        return mask
def kernal_unstructured(module, name):
    KernalPruningMethod.apply(module, name)
    return module


# #### After pruning create new model with updated pruning list

# In[8]: Update feature list by removing prune count for specific value of prune count
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

# In[9]: Copy the non zero weight value from prune model to new model 
def deepCopy(destModel,sourceModel):
    print("Deep Copy Started")
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
                

# In[ ]: Perform Pruning Blockwise For Each Layer of Block
def iterativePruning(newModel,module,blockList,prune_epochs):
    for i in range(prune_epochs):
        # 1.  Initialization: blockList,featureList,convidx,prune_count,module
        # 2.  ComputeCandidateLayer
        candidateConvLayer = computeCandidateConvLayer(module=module,blockList=blockList)
        # 3.  ComputenewList
        newList = computeNewList()
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
                         epochs=1, max_lr=0.01, weight_decay=0, L1=0, grad_clip=.1, logFile=logFile)
        # 10. Evalute the pruned model 
        # 11. Continue another iteration if required and accepted
    
iterativePruning(newModel=newModel,module=module,blockList=blockList,prune_epochs=2)
