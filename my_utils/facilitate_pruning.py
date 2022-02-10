#!/usr/bin/env python
# coding: utf-8

# In[1]:


import my_utils.loadModel as lm
import my_utils.initialize_pruning as ip
import torch


# In[2]:


#t = Tensor to be prune, n is ln normalization, dim dimension over which we want to perform 
def _compute_distance_score(t, n=1, dim_to_keep=[0,1],threshold=1):
        # dims = all axes, except for the one identified by `dim`        
        dim_to_prune = list(range(t.dim()))   #initially it has all dims
        #remove dim which we want to keep from dimstoprune
        for i in range(len(dim_to_keep)):   
            dim_to_prune.remove(dim_to_keep[i])
        
        size = t.shape
        print(f"\nShape of the tensor: {size}")
        print(f"Print the Dims we want to keep: {dim_to_keep}")
        
        module_buffer = torch.zeros_like(t)
                
        #shape of norm should be equal to multiplication of dim to keep values
        norm = torch.norm(t, p=n, dim=dim_to_prune)
        print(f"norm shape = {norm.shape}")
        size = t.shape
        print("Number Of Features Map in current  layer l     =",size[0])
        print("Number Of Features Map in previous layer (l-1) =",size[1])
        
        for i in range(size[0]):
            for j in range(size[1]):
                module_buffer[i][j] = t[i][j]/norm[i][j]
        
        dist = torch.zeros(size[1],size[0],size[0])
        
        channelList = []
        for j in range(size[1]):
            idxtupple = []
            print('.',end='')
            for i1 in range(size[0]):
                for i2 in range((i1+1),size[0]):
                    dist[j][i1][i2] = torch.norm( (module_buffer[i1][j]-module_buffer[i2][j]) ,p=1)
                    dist[j][i2][i1] = dist[j][i1][i2]
                    
                    if dist[j][i1][i2] < threshold:
                        idxtupple.append([j,i1,i2,dist[j][i1][i2]])
            channelList.append(idxtupple)
        return channelList


# In[3]:


def sortKernalByDistance(kernalList):
    for i in range(len(kernalList)):
        iListLen = len(kernalList[i])
        #print(f'lemgth of list {i} ={iListLen}')
        for j in range(iListLen):
            for k in range(iListLen-j-1):
                #print(f"Value of i={i}     Value of j={j} Value of k={k}")
                if kernalList[i][k+1][3] < kernalList[i][k][3]:
                    kernalList[i][k+1], kernalList[i][k] = kernalList[i][k], kernalList[i][k+1]


# In[4]:


def get_k_element(channel_list,k):
    channel_k_list = []
    for i in range(len(channelTuppleList)):
        for j in range(k):
            channel_k_list.append(channel_list[i][j])
    return channel_k_list


# In[5]:


#t = Tensor to be prune, n is ln normalization, dim dimension over which we want to perform 
def _compute_kernal_score(t, n=1, dim_to_keep=[0,1],threshold=1):
        # dims = all axes, except for the one identified by `dim`        
        dim_to_prune = list(range(t.dim()))   #initially it has all dims
        
        #remove dim which we want to keep from dimstoprune
        for i in range(len(dim_to_keep)):   
            dim_to_prune.remove(dim_to_keep[i])
        
        size = t.shape
        print(size)
        print(dim_to_keep)
        
        module_buffer = torch.zeros_like(t)
        #sshape of norm should be equal to multiplication of dim to keep values
        norm = torch.norm(t, p=n, dim=dim_to_prune)
        kernelList = []
        size = norm.shape
        for i in range(size[0]):
            for j in range(size[1]):
                kernelList.append([i,j,norm[i][j]])
            
        return kernelList


# In[6]:


def sofKernelByValue(kernelList):
    return kernelList


# In[7]:


def displayLayer(channelTupple):
    for i in range(len(channelTupple)):
        for j in range(len(channelTupple[i])):
            if j%3==0:
                print()
            print(channelTupple[i][j],'\t',end='')  


# In[8]:


model_name = 'vgg16'
model = lm.load_model(model_name=model_name,number_of_class=6,pretrainval=True)


# In[9]:


blocks = ip.getBlockList(modelname=model_name)
feature_list = ip.getFeatureList(modelname=model_name)
module = ip.getPruneModule(model,prunelist=feature_list)
prune_count = ip.getPruneCount(module=module,blocks=blocks,maxpr=0.25)
print(f"blocks            = {blocks} \n"
      f"feature list      = {feature_list} \n"
      f"prune count list  = {prune_count}")


# In[10]:


channelTuppleList = []
st =2
en = 4
for i in range(st,en):
    channelTuppleList.append(_compute_distance_score(module[i]._parameters['weight'],threshold=1))
print("\n\n\nHere is the :",len(channelTuppleList))


# In[14]:


for i in range(len(channelTuppleList)):
    for j in range(len(channelTuppleList[i])):
        print(f"\n\nlength of list: {len(channelTuppleList[i][j])} and 1st 3 ele are\n{(channelTuppleList[i][j][0:3])}")


# In[15]:


for i in range(len(channelTuppleList)):
    sortKernalByDistance(channelTuppleList[i])


# In[17]:


for i in range(len(channelTuppleList)):
    for j in range(len(channelTuppleList[i])):
        print(f"\n\nlength of list: {len(channelTuppleList[i][j])} and 1st 3 ele are\n{(channelTuppleList[i][j][0:3])}")


# In[ ]:


for i in range(len(channelTuppleList)):
    print("\n\nRow :",i)
    for j in range(3):
        for k in range(len(channelTuppleList[i][j])):
            print(channelTuppleList[i][j][k])


# In[ ]:


newList = []
for i in range(len(channelTuppleList)):
    newList.append(get_k_element(channel_list=channelTuppleList[i],k=prune_count[i]) )

for i in range(len(newList)):
    print(f"\n\n\nlenth of list: {len(newList[i])}")
    for j in range(len(newList[i])):
        
        for k in range(len(newList[i][j])):
            print(newList[i][j][k])


# In[ ]:


for i in range(len(channelTuppleList)):
    print("\n**************************************************************************************************************************")
    displayLayer(channelTuppleList[i])


# In[ ]:




