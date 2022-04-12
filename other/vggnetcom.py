#!/usr/bin/env python
# coding: utf-8

# In[1]:


import my_utils.loadDataset as dl
import my_utils.loadModel as lm
import my_utils.trainModel as tm
import torch


# In[2]:


dl.setFolderLocation(datasets='/home/pragnesh/Dataset/',selectedDataset='IntelIC/',train='train',test='test')
dl.setImageSize(224)
dl.setBatchSize = 2
dataLoader = dl.dataLoader()


# In[3]:


device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device1)
newmodel = lm.loadModel(modelname='vgg16',numberOfClass=6,pretrainval=False,
                        freezeFeature=False,device=device1)


# In[ ]:


print(newmodel.features[0]._parameters)


# In[4]:


logFile = '/home/pragnesh/Dataset/Intel_Image_Classifacation_v2/Logs/ConvModelv2.log'
#tm.device = torch.device('cpu')
tm.fit_one_cycle(dataloaders=dataLoader,trainDir=dl.trainDir,testDir=dl.testDir,
                 ModelName='vgg16',model=newmodel,device_l=device1,
                 epochs=1, max_lr=0.01, weight_decay=0, L1=0, grad_clip=.1, logFile=logFile)


# In[ ]:





# In[ ]:




