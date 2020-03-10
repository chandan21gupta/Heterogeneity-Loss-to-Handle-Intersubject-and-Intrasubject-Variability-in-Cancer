import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import modified
import model as m
from centre_loss import CenterLoss 
from focal_ce_loss import FocalLoss
from train import train_model
from logger import Logger
import copy
from sklearn.metrics import precision_recall_fscore_support
import shutil
import cv2
from PIL import Image


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(360),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


#image_test_dir = 'Transfer/Test/'
#test_image_dataset = modified.MyImageFolder(root=image_test_dir,
#                                          transform=data_transforms['val'])

#test_image_loader = DataLoader(test_image_dataset,batch_size=16, shuffle=False, num_workers=0)
#test_size=len(test_image_dataset)



device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = data_transforms['val'](image)
    image =  image.to(device)
    image = image.unsqueeze(0)
    return image  #assumes that you're using GPU


model=m.inception_v3(pretrained=True)
model.aux_logits=True
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.AuxLogits=m.InceptionAux(768,2)
model = model.to(device)

path='results/'
ckpt_path='models/patient_class_independent_icentre_Adam_relative_alpha_rerun/133_checkpoint.pth.tar'
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['state_dict'])

if(not os.path.exists(path+'{}'.format(ckpt_path.split('/')[-1]))):
    os.makedirs(path+'{}/all'.format(ckpt_path.split('/')[-1]))
    os.makedirs(path+'{}/hem'.format(ckpt_path.split('/')[-1]))

model.eval()
FOLDER_TEST_PATH='Transfer/dhruva_val_set/Folder_Valid/'

for classes in os.listdir(FOLDER_TEST_PATH):
    print(classes)
    for patients in os.listdir(FOLDER_TEST_PATH+classes):
        count=0
        files=os.listdir(FOLDER_TEST_PATH+classes+'/'+patients)
        for f in files:
            image = image_loader(FOLDER_TEST_PATH+classes+'/'+patients+'/'+f)
            x, outputs= model(image)
            _, preds = torch.max(outputs, 1)
            preds=preds.cpu().numpy()
            if(classes=='all' and preds[0]==1):
                count=count+1
                shutil.copyfile(FOLDER_TEST_PATH+classes+'/'+patients+'/'+f,path+'{}/all/'.format(ckpt_path.split('/')[-1])+f)
            elif(classes=='hem' and preds[0]==0):
                count=count+1
                shutil.copyfile(FOLDER_TEST_PATH+classes+'/'+patients+'/'+f,path+'{}/hem/'.format(ckpt_path.split('/')[-1])+f)
        print(patients,count,len(files))

'''
path='Transfer/'
folders=['Folder_fold_0','Folder_fold_1','Folder_fold_2']
all_cells=[]
hem_cells=[]

for folder in folders:
    for classes in os.listdir(path+folder):
        for patients in os.listdir(path+folder+'/'+classes):
            files=os.listdir(path+folder+'/'+classes+'/'+patients)
            if(classes=='all'):
                all_cells.append(len(files))
            else:
                hem_cells.append(len(files))

all_cells.sort(reverse=True)
hem_cells.sort(reverse=True)
print('ALL:')
print(all_cells)
print('HEM')
print(hem_cells)
'''