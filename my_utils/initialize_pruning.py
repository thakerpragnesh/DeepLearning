import torch

'''In[2]: VGGNet is divided into 5 block each will have number of conv layer
             get list of number of conv layer in each block
'''
def getBlockList(modelname):
    blocks = []
    if modelname == 'vgg11':
        blocks    = [1, 1, 2, 2, 2]

    if modelname == 'vgg11bn':
        blocks    = [1, 1, 2, 2, 2]

    if modelname == 'vgg13':
        blocks    = [2, 2, 2, 2, 2]

    if modelname == 'vgg13bn':
        blocks    = [2, 2, 2, 2, 2]

    if modelname == 'vgg16':
        blocks    = [2, 2, 3, 3, 3]

    if modelname == 'vgg16bn':
        blocks    = [2, 2, 3, 3, 3]
    return blocks


'''In[3]: each vgg net parameter is distributed in feature list and classifier list
            get the index where conv layer is present
'''
def getFeatureList(modelname):
    feature_list = []
    
    if modelname == 'vgg11':
        feature_list = [0, 3, 6,8, 11,13, 16,18]

    if modelname == 'vgg11bn':
        feature_list = [0, 3, 6,8, 11,13, 16,18]

    if modelname == 'vgg13':
        feature_list = [0,2, 5,7, 10,12, 15,17, 20,22]

    if modelname == 'vgg13bn':
        feature_list = [0,2, 5,7, 10,12, 15,17, 20,22]

    if modelname == 'vgg16':
        feature_list = [0,2, 5,7, 10,12,14, 17,19,21, 24,26,28]

    if modelname == 'vgg16bn':
        feature_list = [0,2, 5,7, 10,12,14, 17,19,21, 24,26,28]
    return feature_list


''' In[4]: extract the feature and store in the list for convinience in use
'''
def getPruneModule(newModel,prunelist):
    module = []
    for i in prunelist:
        module.append(newModel.features[i])
    return module        


'''In[5]: we can prune different number of kernel from different list
in the following code we are distributing pruning prob block wise.
Here earlier block have less prune prob and later have more prune prob
and last block has prune prob equals to maxpr provided in the function  
'''
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
