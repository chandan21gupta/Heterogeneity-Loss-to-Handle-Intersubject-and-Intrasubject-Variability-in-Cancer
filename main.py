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
import argparse
from logger import Logger

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


parser = argparse.ArgumentParser(description='Basic Inputs')
parser.add_argument('--train_dir', type=str, help='Train dir of images')
parser.add_argument('--valid_dir', type=str, help='Valid dir of images')
parser.add_argument('--test_dir', type=str, help='Test dir of images')
parser.add_argument('--exp_name', type=str, help='Experiment name to save models and Tensorboard_logs')


args = parser.parse_args()


image_train_dir = args.train_dir
image_valid_dir = args.valid_dir
image_test_dir = args.test_dir


train_image_dataset = modified.MyImageFolder(root=image_train_dir,
                                           transform=data_transforms['train'])

valid_image_dataset = modified.MyImageFolder(root=image_valid_dir,
                                           transform=data_transforms['val'])

test_image_dataset = modified.MyImageFolder(root=image_test_dir,
                                          transform=data_transforms['val'])


train_image_loader = DataLoader(train_image_dataset,batch_size=16, shuffle=True, num_workers=32)
valid_image_loader = DataLoader(valid_image_dataset,batch_size=256, shuffle=False, num_workers=32)
test_image_loader = DataLoader(test_image_dataset,batch_size=256, shuffle=False, num_workers=32)

train_size=len(train_image_dataset)
valid_size=len(valid_image_dataset)
test_size=len(test_image_dataset)


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
centres_dict={}
centres_type_dict={}

count=0


for types in os.listdir(image_train_dir):
    files=os.listdir(image_train_dir+'/'+types)
    for f in files:
        patient_name='_'.join(f.split('_')[:2])
        if(patient_name in centres_dict.keys()):
            continue
        else:
            centres_dict[patient_name]=count
            if(types=='all'):
                centres_type_dict[count]=0
            else:
                centres_type_dict[count]=1

            count=count+1

print(centres_type_dict,count)
class_patients=[]
for i in range(count):
    class_patients.append(centres_type_dict[i])

class_centres=[0,1]

print(device,centres_dict,len(centres_dict.keys()),centres_type_dict,class_patients)

model=m.inception_v3(pretrained=True)
model.aux_logits=True

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model.AuxLogits=m.InceptionAux(768,2)

model = model.to(device)

print(class_centres)

center_loss_1 = CenterLoss(patients_class=class_patients, num_classes=len(centres_type_dict.keys()), feat_dim=2048, device=device ,use_gpu=True)
center_loss_2 = CenterLoss(patients_class=class_centres, num_classes=2, feat_dim=2048, device=device ,use_gpu=True)

focal_cross_loss=FocalLoss(class_num=2,device=device)

optim_1 = optim.Adam(model.parameters(), lr=0.001, betas=(0.8, 0.99), amsgrad=True)
optim_2 = optim.Adam(center_loss_1.parameters(), lr=0.5, betas=(0.8, 0.99), amsgrad=True)
optim_3 = optim.Adam(center_loss_2.parameters(), lr=0.5, betas=(0.8, 0.99), amsgrad=True)

scheduler_1 = lr_scheduler.MultiStepLR(optim_1, milestones=[15,25,35,45,55,65,75,85], gamma=0.5)
scheduler_2 = lr_scheduler.MultiStepLR(optim_2, milestones=[14,21,28,35,42,49], gamma=0.5)
scheduler_3 = lr_scheduler.MultiStepLR(optim_3, milestones=[14,21,28,35,42,49], gamma=0.5)


#scheduler_1 = lr_scheduler.MultiStepLR(optim_1, milestones=[15,25,35,45,55], gamma=0.5)
#scheduler_2 = lr_scheduler.MultiStepLR(optim_2, milestones=[14,21,28,35,42,49], gamma=0.5)

epochs=1000
alpha =0.1
model_name=args.exp_name

log_save_path='logs/{}/'.format(model_name)
if(not os.path.exists(log_save_path)):
    os.makedirs(log_save_path)
    os.makedirs('models/{}'.format(model_name))
    os.makedirs('models_stats/{}'.format(model_name))

logger = Logger(log_save_path)
print('Going for training')

train_model(model=model, centre_dict=centres_dict, centre_type_dict=centres_type_dict,
            Center_loss_1=center_loss_1, Center_loss_2=center_loss_2, Focal_loss=focal_cross_loss,
            optimizer_1=optim_1, optimizer_2=optim_2 , optimizer_3=optim_3, num_epochs=epochs, train_loader=train_image_loader,
            valid_loader=valid_image_loader, device=device, alpha=alpha, t_size=train_size,
            v_size=valid_size, te_size=test_size, logger=logger, test_loader=test_image_loader,
            model_name=model_name, sched_1=scheduler_1, sched_2=scheduler_2, sched_3=scheduler_3)


