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
import copy
from sklearn.metrics import precision_recall_fscore_support
import shutil

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'best_model_file.pth.tar')

def metrics(pred,true):
    precision,recall,f1,support= precision_recall_fscore_support(true, pred,labels=[0,1])
    return precision[0],precision[1],recall[0],recall[1],f1[0],f1[1]

def train_model(model, centre_dict, centre_type_dict, Center_loss_1, Center_loss_2, 
                Focal_loss, optimizer_1, optimizer_2 , optimizer_3, num_epochs, 
                train_loader, valid_loader, device, alpha, t_size, v_size, te_size, logger, 
                test_loader, model_name, sched_1,sched_2,sched_3):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_test_acc=0.0
    #temp_list=[]

    for epoch in range(num_epochs):
        epoch_time=time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                sched_1.step()
                sched_2.step()
                sched_3.step()

                dataloader= train_loader
                model.train()  # Set model to training mode

            else:
                dataloader= valid_loader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            total_preds=[]
            total_labels=[]

            # Iterate over data.
            if(phase=='train'):
                for inputs, labels, paths in dataloader:
                    
                    temp_list=[]
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    #print(inputs.size(),labels.size())
                    
                    for path in paths:
                        image_name=path.split('/')[-1]
                        temp_list.append(centre_dict['_'.join(image_name.split('_')[:2])])
                    
                    centre_labels= torch.from_numpy(np.asarray(temp_list)).to(device)

                
                    patient_centre_features=[]
                    for c in centre_labels:
                        #print(c.item())
                        patient_centre_features.append(Center_loss_1.return_center(c.item()))


                    patient_centre_features = torch.stack(patient_centre_features)
                    
                    
                    #print(patient_centre_features.size())
                    #print(patient_centre_features)
                    #print(centre_labels,type(centre_labels))

                    # zero the parameter gradients
                    optimizer_1.zero_grad()
                    optimizer_2.zero_grad()
                    optimizer_3.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        x, outputs,aux= model(inputs)
                        _, preds = torch.max(outputs, 1)
                    
                        loss_aux = Focal_loss(aux,labels)

                        loss_1 = Center_loss_1(x , centre_labels, labels, 2)
                        loss_3 = Center_loss_2(x , labels, labels, 1) 
                        #print(loss_1)
                        loss_2 = Focal_loss(outputs,labels)
                        #print(loss_2)
                        
                        loss = loss_3 * (alpha/10) +loss_1 * alpha + loss_2 + (0.4 * loss_aux)
 
                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer_1.step()
                                               
                        for param in Center_loss_1.parameters():    
                            param.grad.data *= (1./alpha)

                        for param in Center_loss_2.parameters():    
                            param.grad.data *= (1./(alpha/10))

                        optimizer_2.step()
                        optimizer_3.step()

                        total_preds.extend(preds.cpu().numpy())
                        total_labels.extend(labels.data.cpu().numpy())
                        
                    # statistics
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / t_size
                epoch_acc = running_corrects.double() / t_size

                train_loss=epoch_loss 
                train_acc=epoch_acc
                print('SHAPES: ',len(total_preds))
                train_precision_class0,train_precision_class1,train_recall_class0,train_recall_class1, train_f1_class0, train_f1_class1=metrics(np.asarray(total_preds),np.asarray(total_labels))

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                #np.save('models_stats/{}'.format(model_name)+"/train_epoch_"+str(epoch)+"_pred.npy",preds.cpu().numpy())
                #np.save('models_stats/{}'.format(model_name)+"/train_epoch_"+str(epoch)+"_true.npy",labels.data.cpu().numpy())

            if(phase=='val'):
                for inputs, labels, paths in dataloader:
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        x, outputs= model(inputs)
                        _, preds = torch.max(outputs, 1)
                   
                        loss= Focal_loss(outputs,labels)

                        # backward + optimize only if in training phase

                        total_preds.extend(preds.cpu().numpy())
                        total_labels.extend(labels.data.cpu().numpy())

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / v_size
                epoch_acc = running_corrects.double() / v_size


                val_loss=epoch_loss 
                val_acc=epoch_acc
                print('SHAPES: ',len(total_preds))

                val_precision_class0,val_precision_class1,val_recall_class0,val_recall_class1, val_f1_class0, val_f1_class1=metrics(np.asarray(total_preds),np.asarray(total_labels))

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model_{}.pth.tar'.format(epoch))

        info = {


                'Loss_1_Patient_Centre_Loss':  loss_1,
                'Loss_2_Cross_Entropy_Loss':   loss_2,
                'Loss_3_Patient_Class_Centre_Loss': loss_3,
                'validation_loss':             val_loss,
                'validation_accuracy':         val_acc,

                'validation_precision_class0': val_precision_class0,
                'validation_precision_class1': val_precision_class1,
                'validation_recall_class0':    val_recall_class0,
                'validation_recall_class1':    val_recall_class1,
                'validation_f1_class0':        val_f1_class0,
                'validation_f1_class1':        val_f1_class1,


                'train_loss': train_loss,
                'train_accuracy': train_acc,

                'train_precision_class0': train_precision_class0,
                'train_precision_class1': train_precision_class1,
                'train_recall_class0': train_recall_class0,
                'train_recall_class1': train_recall_class1,
                'train_f1_class0': train_f1_class0,
                'train_f1_class1': train_f1_class1
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch+1)
        print('TIME TAKEN FOR EPOCH: ',time.time()-epoch_time)

        #test_loss,test_acc,test_precision_class0,test_precision_class1,test_recall_class0,test_recall_class1,test_f1_class0,test_f1_class1 = test_model(model,test_loader,device,logger,te_size,epoch,Focal_loss,model_name)
            
        #is_best = test_acc > best_test_acc
        #best_test_acc = max(test_acc, best_test_acc)
        '''
        if (epoch%2==0):

            save_checkpoint({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_test_acc': best_test_acc,

                'test_loss':             test_loss,
                'test_accuracy':         test_acc,

                'test_precision_class0': test_precision_class0,
                'test_precision_class1': test_precision_class1,
                'test_recall_class0':    test_recall_class0,
                'test_recall_class1':    test_recall_class1,
                'test_f1_class0':        test_f1_class0,
                'test_f1_class1':        test_f1_class1,


                'train_loss': train_loss,
                'train_accuracy': train_acc,

                'train_precision_class0': train_precision_class0,
                'train_precision_class1': train_precision_class1,
                'train_recall_class0': train_recall_class0,
                'train_recall_class1': train_recall_class1,
                'train_f1_class0': train_f1_class0,
                'train_f1_class1': train_f1_class1,

                'optimizer_1' : optimizer_1.state_dict(),
                'optimizer_2' : optimizer_2.state_dict(),
                'optimizer_3' : optimizer_3.state_dict(),

            }, is_best, filename= "models/{}/".format(model_name)+str(epoch+1)+"_checkpoint.pth.tar")
        '''

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    #print('Best test Acc: {:4f}'.format(best_test_acc))
    
    '''
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    '''

def test_model(model,test_loader,device,logger,te_size,epoch,Focal_loss,model_name):
    
    running_loss = 0.0
    running_corrects = 0
    model.eval()

    total_preds=[]
    total_labels=[]

    for inputs, labels, paths in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            x, outputs= model(inputs)
            _, preds = torch.max(outputs, 1)
            
            loss= Focal_loss(outputs,labels)

            total_labels.extend(labels.data.cpu().numpy())
            total_preds.extend(preds.cpu().numpy())
            # # statistics
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / te_size
    epoch_acc = running_corrects.double() / te_size

    test_loss=epoch_loss 
    test_acc=epoch_acc
    print('SHAPES: ',len(total_preds))

    test_precision_class0,test_precision_class1,test_recall_class0,test_recall_class1, test_f1_class0, test_f1_class1=metrics(np.asarray(total_preds),np.asarray(total_labels))

    info = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_precision_class0': test_precision_class0,
        'test_precision_class1': test_precision_class1,
        'test_recall_class0': test_recall_class0,
        'test_recall_class1': test_recall_class1,
        'test_f1_class0': test_f1_class0,
        'test_f1_class1': test_f1_class1
        }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch+1)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', test_loss, test_acc))

    np.save('models_stats/{}'.format(model_name)+"/test_epoch_"+str(epoch)+"_pred.npy",preds.cpu().numpy())
    np.save('models_stats/{}'.format(model_name)+"/test_epoch_"+str(epoch)+"_true.npy",labels.data.cpu().numpy())

    return test_loss,test_acc,test_precision_class0,test_precision_class1,test_recall_class0,test_recall_class1,test_f1_class0,test_f1_class1
