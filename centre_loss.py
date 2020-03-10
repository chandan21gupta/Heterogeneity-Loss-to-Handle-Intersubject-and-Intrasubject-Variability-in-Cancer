import torch
import torch.nn as nn
import numpy as np

class CenterLoss(nn.Module):

    """
    Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    
    def __init__(self,num_classes, patients_class, feat_dim, device, contra_margin=2048 ,eps=1e-9, use_gpu= True):

        super(CenterLoss, self).__init__()
        self.patients_class = patients_class
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.num_classes=num_classes
        self.contra_margin=contra_margin
        self.eps = eps 
        self.device=device

        if self.use_gpu: #try randn too
            self.centers = nn.Parameter(torch.rand(self.num_classes, self.feat_dim).to(self.device))
            #self.class_centers=nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.rand(self.num_classes, self.feat_dim))
            #self.class_centers=nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    
    def contra(self, output1, output2, target, size_average=True):

        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        
        return losses.mean() if size_average else losses.sum()

    
    def triplet(self, anchor, positive, negative, size_average=True):

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


    def image_centre_loss(self,x,labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: 
            classes = classes.to(self.device)
        
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        #
        # (mask)
        #print(type(mask))
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)

        #print(dist,type(dist))

        dist = torch.cat(dist)
        loss = dist.mean()
        return loss

    def class_centre_loss(self, labels_1, labels_2):        
        loss_intra=[]
        loss_inter=[]
        for i in range(len(labels_1)):
            for j in range(len(labels_1)):
                #val=val.item()
                val=((self.centers[labels_1[i]]-self.centers[labels_1[j]])**2).mean()
                val = val.clamp(min=1e-12, max=1e+12) # for numerical stability
                #print(val)
                val =torch.unsqueeze(val,0)
                #print(val)
                if(labels_2[i]==labels_2[j]):
                    loss_intra.append(val)
                else:
                    loss_inter.append(1/(1+val))

        #print(len(loss_inter),len(loss_intra))
        #print(loss_inter,loss_intra)
        #print(type(loss_inter),type(loss_intra))
        #loss_inter=torch.from_numpy(np.asarray(loss_inter)).to(self.device)
        #loss_intra=torch.from_numpy(np.asarray(loss_intra)).to(self.device)
        
        if(len(loss_inter)!=0 and len(loss_intra)!=0):
            loss_inter=torch.cat(loss_inter)
            loss_inter=loss_inter.mean()            
            loss_intra=torch.cat(loss_intra)
            loss_intra=loss_intra.mean()
            return loss_intra+loss_inter

        if(len(loss_inter)!=0):
            loss_inter=torch.cat(loss_inter)
            loss_inter=loss_inter.mean()
            return loss_inter
        
        if(len(loss_intra)!=0):
            loss_intra=torch.cat(loss_intra)
            loss_intra=loss_intra.mean()
            return loss_intra 
    
    def return_center(self,index):
        return self.centers[index]

    def forward(self, x, labels, labels_2, y):
        '''
        loss_1= self.image_centre_loss(x,labels)
        loss_2= self.class_centre_loss()
        
        return loss_1 + loss_2  
        '''
        if(y==1):
            return self.image_centre_loss(x,labels)
        else:
            return self.image_centre_loss(x,labels)+self.class_centre_loss(labels,labels_2)
