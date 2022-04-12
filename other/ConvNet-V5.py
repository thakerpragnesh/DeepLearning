#!/usr/bin/env python
# coding: utf-8

#In[1 ]:Define function to ensure all output location exist properly
import os
def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


######################################################################################################################################################################
# In[2]: Set all the file path for all neccesory things
DatasetLoc = '/home3/pragnesh/Dataset/'
SelDataSet = 'cifar10'
trainDir = 'train'
testDir = 'test'
SelectOutLoc = DatasetLoc+"cifar10_Out/"  #/home3/pragnesh/Dataset/Intel_Image_Classifacation_v2/
zipFile = False
ifLoadModel = False
modelname ='vgg16' # vgg11 vgg13 vgg16,  resnet18,  savedmodel

data_dir = DatasetLoc+SelDataSet
LogLoc =   SelectOutLoc+"Logs/"                          #/home3/pragnesh/Dataset/Intel_Image_Classifacation_v2/Logs/
ModelLoc = SelectOutLoc+"Model/"
SavePath = SelectOutLoc+'Model/VGG_IntelIC_v1-'
LoadPath = SelectOutLoc+'Model/VGG_IntelIC_v1-'
outfile = LogLoc+"FinalOutv2.log"                       #/home3/pragnesh/Dataset/Intel_Image_Classifacation_v2/FinalOutv2.log
logFile = LogLoc+"ConvModelv2.log"                      #/home3/pragnesh/Dataset/Intel_Image_Classifacation_v2/ConvModelv2.log
TransferLearning = False
NumberOfClass = 10


######################################################################################################################################################################
# In[3]: Make sure all output directories existed
ensure_dir(SelectOutLoc)
ensure_dir(ModelLoc)
ensure_dir(LogLoc)



######################################################################################################################################################################
# In[4]:
#import basic library for some basic function
import numpy as np
"""
Following Cell facilitate to read files from drive and and help to 
read dataset
"""
# import library to perform file operation
import os #use to access the files 
import tarfile # use to extract dataset from zip files
import sys
import zipfile
"""
Torch Library provides facilities to create networl architechture and 
write farword and backwor phase od neural network
"""
#import torch library to build neural network
import torch  # Elementory function of tensor is define in torch package
import torch.nn as nn # Several layer architectur is define here
import torch.nn.functional as F # loss function and activation function



######################################################################################################################################################################
# In[5]:
"""
Computer vision is one of the most important application and thus lots 
of deplopment in the and torch.vision provides many facilities that can 
be use to imporve model such as data augmentation, reading data batchwise, 
suffling data before each epoch and many more
"""
# import torch library related to image data processing
import torchvision # provides facilities to access image dataset
from torchvision.datasets.utils import download_url 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision import datasets, models, transforms



######################################################################################################################################################################
# In[6]: Load data from zip file into selceted location for first time execution
"""#Data Prepration 1
Based on the image size of the dataset choose apropriate values of the color channel and Image Size
Here we can define path to a folder where we can keep all the dataset. 
In the following we are using the zip files. Originally dataset should 
be in the following format DataSetName is parent folder and it should 
contain train and test folder. train and test folder should contain 
folder for each category and images of respective category should be in 
the respective category folder
"""
######################### Data Loading #########################################
if zipFile == True:
  fullpath = DatasetLoc+SelDataSet+'.zip'
  zip_ref = zipfile.ZipFile(fullpath, 'r') #Opens the zip file in read mode
  zip_ref.extractall('/tmp') #Extracts the files into the /tmp folder
  zip_ref.close()



######################################################################################################################################################################
# In[7]: Create batch of dataset and perform data transformation
"""
Choose an apropriate batch size that can be loaded in the current 
enviroment without crashing and also do not choose too big batch even 
if dataset is small because it leads to very few updates per epoch
"""
#################### Create Batch Of Dataset and do data augmentation ###########
bs = 128
ImageSize = 224

"""
Data Augmentaion generally help in reducing overfitting error during 
trainng process and thus we are performing randon horizontal flip and 
random crop during training but during validation as no training happens 
we dont perform data augmentation
"""
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    trainDir: transforms.Compose([
        transforms.RandomResizedCrop(ImageSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    testDir: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in [trainDir, testDir]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs,
                                             shuffle=True, num_workers=1)
              for x in [trainDir, testDir]}

dataset_sizes = {x: len(image_datasets[x]) for x in [trainDir, testDir]}
class_names = image_datasets[trainDir].classes



######################################################################################################################################################################
# In[8]: Check for cuda enabled device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""Check if Cuda GPU is available"""
#check for CUDA enabled GPU card
def getDeviceType():
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')
device = getDeviceType()
print(device)


######################################################################################################################################################################
# In[9]: Define Function te facilitate training process such as compute loss and deiplay result
"""#Training Process
Below code will work as a base function and provide all the important 
function like compute loss, accuracy and print result in a perticular 
formate afte each epoch. Funvtion are as follow
1. Accuracy : Computer accuracy in evalutaion mode of pytorch on given dataset for given model
2. compute_batch_loss : Compute batch loss and append the loss in the list of batch loss.
3. compute_batch_loss_acc : Compute batch loss, batch accuracy and append the loss in the list of batch loss.
4. accumulate_batch_loss_acc: Accumulate loss from the list of batch and acccuraly loss.
5. Epoch end to print the output after every epoch in proper format
"""
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1) 		# get the prediction vector
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Compute loss of the given batch and return it
def compute_batch_loss(newmodel, batch_X,batch_y):
  images = batch_X.to(device)
  labels = batch_y.to(device)
  out = newmodel(images)                  		# Generate predictions
  loss = F.cross_entropy(out, labels) 			# Calculate loss
  return loss

# Computes loss and accuracy of the given batch(Used in validation)
def compute_batch_loss_acc(newmodel, batch_X,batch_y):
    images = batch_X.to(device)
    labels = batch_y.to(device)
    out = newmodel(images)                    	# Generate predictionsin_features=4096
    loss = F.cross_entropy(out, labels)   		# Calculate loss
    acc = accuracy(out, labels)           		# Calculate accuracy
    return {'val_loss': loss, 'val_acc': acc}

# At the end of epoch accumulate all batch loss and batch accueacy    
def accumulate_batch_loss_acc(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def epoch_end(epoch, result):
  # Print in given format 
  # Epoch [0], last_lr: 0.00278, train_loss: 1.2862, val_loss: 1.2110, val_acc: 0.6135
  strResult = "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
      epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc'])
  print(strResult)
  return strResult



######################################################################################################################################################################
# In[10]: define training process
"""## Define Training 
Here we will evalute our model after each epoch on validation dataset using evalute method
get_lr method returnd last learning rate used in the training
Here we are using one fit cycle method in which we specify the max learning rate and learning 
rate start from 1/10th value of max_lr and slowly increases the value to max_lr for 40% of updates 
then decreases to its initial value for 40% updates and then further decreases to 1/100th of max_lr 
value to perform final fine tuning.
"""
# evalute model on given dataset using given data loader
@torch.no_grad()
# evalute model on given dataset using given data loader
def evaluate(model, data_loader):
    model.eval()
    with torch.no_grad():
      for batch_X, batch_y in data_loader:
        outputs = [compute_batch_loss_acc(model,batch_X,batch_y)]
      return accumulate_batch_loss_acc(outputs)

# Use special scheduler to change the value of learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# epoch=8, max_lr=.01, weight_decay(L2-Regu parametr)=.0001,opt_func=Adam

######### Main Function To Implement Training #################
def fit_one_cycle(ModelName,epochs, max_lr, model, 
                  weight_decay=0, L1=0,grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    # Set up cutom optimizer here we will use one cycle scheduler with max learning
    # rate given by max_lr, default optimizer is SGD but we will use ADAM, and 
    # L2 Regularization using weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(dataloaders[trainDir]))
    print("Training Starts")
    with open(logFile, "a") as f:
      for epoch in range(epochs):
          # Training Phase 
          model.train()  #######################
          train_losses = []
          lrs = []
          #for batch in train_loader:
          for batch_X, batch_y in dataloaders[trainDir]:
              # computer the training loss of current batch
              loss = compute_batch_loss(model,batch_X,batch_y)
              l1_crit = nn.L1Loss()
              reg_loss = 0
              for param in model.parameters():
                reg_loss += l1_crit(param,target=torch.zeros_like(param))
              loss += L1*reg_loss 
              
              train_losses.append(loss)
              loss.backward() # compute the gradient of all weights
              # Clip the gradient value to maximum allowed grad_clip value
              if grad_clip: 
                  nn.utils.clip_grad_value_(model.parameters(), grad_clip)
              optimizer.step() # Updates weights 
              # pytorch by default accumulate grade history and if we dont want it
              # we should make all previous grade value equals to zero
              optimizer.zero_grad() 
              # Record & update learning rate
              lrs.append(get_lr(optimizer))
              sched.step() # Update the learning rate
              # Compute Validation Loss and Valodation Accuracy
              result = evaluate(model, dataloaders[testDir])
              # Compute Train Loss of whole epoch i.e mean of loss of batch 
              result['train_loss'] = torch.stack(train_losses).mean().item()
              # Observe how learning rate is change by schedular
              result['lrs'] = lrs
              # print the observation of each epoch in a proper format
          
          #strResult = "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, 
            #val_acc: {:.4f}".format(epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc'])
          strResult = epoch_end(epoch, result) 
          
          f.write(f"{ModelName}-\t{strResult}\n")
          print(strResult)
          history.append(result) # append tupple result with val_acc, vall_loss, and trin_loss
        
    return history




######################################################################################################################################################################
# In[11]: Load the apropiate model for training
############################# Use Pre-Train Model for Transfer Learning #####################################
#################### Download Pretrain VGGNet ################
if ifLoadModel == False:
    if modelname == 'vgg16' or modelname == 'vgg13' or modelname == 'vgg11':
        if modelname == 'vgg11':
            newModel = torchvision.models.vgg11(pretrained=False)
        if modelname == 'vgg13':
            newModel = torchvision.models.vgg13(pretrained=False)
        if modelname == 'vgg16':
            newModel = torchvision.models.vgg16(pretrained=False)    
        for param in newModel.parameters():
            param.requires_grad = False
        #Need to change the below code if we choose different model
        print(newModel.classifier[6])
        num_ftrs = newModel.classifier[6].in_features
        # Here the size of each output sample is set to 10.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        newModel.classifier[6] = nn.Linear(num_ftrs, NumberOfClass)
        newModel = newModel.to(device)

    ##################### Download Pretrain ResNet 18 ############################
    if modelname == 'resnet18':
        newModel = torchvision.models.resnet18(pretrained=False)
        print(newModel.fc)
        for param in newModel.parameters():
            param.requires_grad = False
        #print(newModel)
        #Need to change the below code if we choose different model
        num_ftrs = newModel.fc.in_features
        # Here the size of each output sample is set to 10.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        newModel.fc = nn.Linear(num_ftrs, 10)
        newModel = newModel.to(device)

################# Load Stored Trained Model #####################################
if ifLoadModel == True:
    if device== torch.device('cpu'):
        newModel = torch.load(LoadPath, map_location=torch.device('cpu'))
    else:
        newModel = torch.load(LoadPath, map_location=torch.device('cuda'))
    # Need to change the following loop for dofferent model
    count=0
    if modelname == 'VGG16':
        for param in newModel.parameters():
            if count in (26,28,30):
              param.requires_grad=True
            else:
              param.requires_grad=False
            
    
    if modelname == 'VGG13':
        for param in newModel.parameters():
            if count in (20,22,24):
              param.requires_grad=True
            else:
              param.requires_grad=False
            
    if modelname == 'VGG11':
        for param in newModel.parameters():
            if count in (16,18,20):
              param.requires_grad=True
            else:
              param.requires_grad=False
print(newModel)



######################################################################################################################################################################
# In[12]: Set Hyper Parameter
"""Set all the Hyper Parameter Such as 
1. Learning Rate to control step size
2. grad_clip to control the maximum value of gradient
3. weight decay to control L2 regularization
4. L1 to control L1 regularization
5. opt_func to select optimization function
"""
# We are using one cycle fit function in which learning rate start with 1/10th 
# of selected maximum learning rate and increase learning rate from min to max
# in 1st phase and then decrease from max to min in 2nd phase
max_lr = 5e-3
epochs = 30
grad_clip = 0.2 # maximum allowed value of grad: grad = min(grad,0.01)
weight_decay = 1e-4 # weight decay is a constant multiplier for L2 regularization
L1 = 1e-5
opt_func = torch.optim.Adam# Choosing Adma optimizer as try to implement SGD+Momentum+Adaptive Learining
MODEL_NAME = f"VGG_Net-\t MLR-{max_lr}-GC{grad_clip}-WD-{weight_decay}-L1-{L1}"



######################################################################################################################################################################
# In[13]: Train the Model
"""Start the training using one fit cycle algorithms where learning rate start with one tenth of provided max_lr and then increase it value till that point and then decrease onword and for last few epoch learning rate furthe decrease
Pass the hyperparameter for training of the model and start training process for set number of epcoh
"""
# Commented out IPython magic to ensure Python compatibility.
# %%time
historylast=[]
historylast += fit_one_cycle(MODEL_NAME,epochs, max_lr, newModel,  
                              grad_clip=grad_clip, 
                              weight_decay=weight_decay, L1=L1
                              ,opt_func=opt_func
                              )
################################################ Evalute the Training Process by Plotting Graph ###################################
with open(outfile,'a') as f:
    f.write("############### Training End Here ########### \n")
torch.save(newModel, SavePath)


######################################################################################################################################################################
# In[14]: Print Accuracy od all epoch
accuracies = [x['val_acc']*100 for x in historylast]
accuracies = ["%.2f" % v for v in accuracies]
print(accuracies)


######################################################################################################################################################################
# In[15]: Define method to compute accuracy for a given model on given dataset
"""##Define method to compute accuracy for a given model on given dataset"""
def accuraciesTotal(newModel, data_loader):
  with torch.no_grad():
    acc = []
    for batch in data_loader:
        images, label = batch
        images, labels = batch[0].to(device), batch[1].to(device)
        out = newModel(images)
        acc.append(accuracy(out, labels))
    return torch.mean(torch.stack(acc))



######################################################################################################################################################################
# In[16]: Load Train and Test Data with same tranformation to evelute train and test error
"""###Prepare the data loader for inference. During the inference we want to perform same tranformation on test and train dataset"""
image_datasets_eval = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[testDir])
                  for x in [trainDir, testDir]}
dataloaders_eval = {x: torch.utils.data.DataLoader(image_datasets_eval[x], batch_size=bs,
                                             shuffle=False, num_workers=1)
                  for x in [trainDir, testDir]}



######################################################################################################################################################################
# In[17]: Evalutute train and test error
trainacc = accuraciesTotal(newModel,dataloaders_eval[trainDir])
testacc = accuraciesTotal(newModel,dataloaders_eval[testDir])

with open(outfile,'a') as f:
    f.write(f"Train Accuracy :  {trainacc}\n Test Accuracy  :  {testacc}")



######################################################################################################################################################################
############################################################# Pruning Start Here #####################################################################################
######################################################################################################################################################################
# In[18]:
import torch.nn.utils.prune as prune
import time

###Select the features want to prune and store them in the list of of Mudule here"""
#print(newModel.features)
Module = []
if modelname == 'vgg11':
    prunelist = [0, 3, 6,8, 11,13, 16,18]
if modelname == 'vgg13':
    prunelist = [0,2, 5,7, 10,12, 15,17, 20,22]
if modelname == 'vgg16':
    prunelist = [0,2, 5,7, 10,12,14, 17,19,21, 24,26,28]

pr = [] 
maxpr=.05
length = len(prunelist)
print(length)
count=0
for i in prunelist:
    Module.append(newModel.features[i])
    frac = length-count
    pr.append(maxpr/frac)
    with open(outfile,'a') as f:
         f.write("Global Sparsity: {:.2f}%".format(pr[count]*100))
    count+=1

"""Select the amount of feature we want to prune in each Layer"""
epochs = 10
max_lr = 1e-3
grad_clip = .2 
weight_decay = 1e-5 
L1=1e-5
itteration = 5
for  ittr in range(itteration):
    for i in range(len(Module)):
      prune.ln_structured(Module[i], name="weight", amount=pr[i], n=1, dim=0)
    numberOfZero = 0
    numberOfElements = 0
    totalNumberOfZero=0
    totalNumberOfElements=0
    for i,j in zip(range(len(prunelist)),prunelist):
        numberOfZero = torch.sum(Module[i].weight == 0)
        totalNumberOfZero += numberOfZero
        numberOfElements = Module[i].weight.nelement()
        totalNumberOfElements += numberOfElements
        frac = 100. * float(torch.sum(Module[i].weight == 0))/float(Module[i].weight.nelement())
        with open(outfile,'a') as f:
            f.write(f"\n {j} Sparsity in {Module[i]} is \t{frac}")
        
    with open(outfile,'a') as f:
        f.write("\n Global Sparsity: {:.2f}% \n\n".format(100*float(numberOfZero)/float(numberOfElements)))

    print(f"******************** {ittr}**********************")
    #fit_one_cycle(ModelName,epochs, max_lr, model,weight_decay=0, L1=0,grad_clip=None, opt_func=torch.optim.SGD):
    historyPrun = fit_one_cycle(MODEL_NAME,epochs, max_lr, newModel,  
                              grad_clip=grad_clip, 
                              weight_decay=weight_decay, L1=L1,
                              opt_func=opt_func
                              )
    
epochs =30
historyPrun = fit_one_cycle(MODEL_NAME,epochs, max_lr, newModel,
              grad_clip=grad_clip,
              weight_decay=weight_decay, L1=L1,
              opt_func=opt_func
              )


for i,j in zip(prunelist,range(len(prunelist))):
    prune.remove(newModel.features[i], 'weight')
torch.save(newModel, "/home3/pragnesh/Model/PruneModel")

with open(outfile,'a') as f:
    f.write("\n execution completed successfully")
######################################################################################################################################################################

