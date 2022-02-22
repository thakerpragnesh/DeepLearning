#!/usr/bin/env python
# coding: utf-8

# In[1]:
import my_utils.loadModel as lm
import my_utils.initialize_pruning as ip
import torch


# In[2]:
#t = Tensor to be prune, n is ln normalization, dim dimension over which we want to perform 
def compute_distance_score(t, n=1, dim_to_keep=[0,1],threshold=1):
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
def sort_kernel_by_distance(kernalList):
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
    for i in range(len(channel_list)):
        tempList = []
        for j in range(k):
            tempList.append(channel_list[i][j])
        channel_k_list.append(tempList)
    return channel_k_list

# In[5]:
#t = Tensor to be prune, n is ln normalization, dim dimension over which we want to perform 
def compute_kernal_score(t, n=1, dim_to_keep=[0,1],threshold=1):
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
def sort_kernel_by_value(kernelList):
    return kernelList


# In[7]:
def displayLayer(channelTupple):
    for i in range(len(channelTupple)):
        for j in range(len(channelTupple[i])):
            if j%3==0:
                print()
            print(channelTupple[i][j],'\t',end='')  
