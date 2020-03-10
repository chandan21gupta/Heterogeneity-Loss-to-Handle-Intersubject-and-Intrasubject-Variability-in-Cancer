import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, class_num, device, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)

        self.gamma = gamma
        self.device= device
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)

        ce_loss=F.cross_entropy(inputs, targets)
                
        return ce_loss      
        '''
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)

        class_mask.scatter_(1, Variable(ids.data), 1.)
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()


        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        
        #batch_loss = batch_loss * class_mask[:,0]
        
        #print(batch_loss)
        
        #count=0
        #for i in range(len(targets)):
        #    if(targets[i].data[0]==0):
        #        count+=1
        #        batch_loss[i]=0.0
        #print('Hello ',count)
                
        #print(batch_loss)
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        
        final_loss=loss+ce_loss # We can check softmax loss too. Has been highly used too
        
        return final_loss
        '''