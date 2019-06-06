#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F




class PositionSensitiveScoreMap(nn.Module):
    
    # inputs to the forward pass ar the classification convolution layer outputs
    # and rois
    # Outputs: scores for all C+1 and for each bbox in the rois 
    
    # Here we define our network structure
    def __init__(self,feat_stride = (520//17)):
        super(PositionSensitiveScoreMap, self).__init__()
        self.feat_stride = feat_stride
        k=7
        self.k=k
        self.softmax = nn.Softmax()
        self.AvgPool = nn.AvgPool2d((7, 7),(7,7))
        
        
    # Here we define one forward pass through the network
    def forward(self, cls_conv_out,rois):
        feat_stride = self.feat_stride
        rois = rois // feat_stride
        k=self.k
        n = rois.shape[0] #number of regions to look at
    
        _,c_plus_1_times_kk,hh,ww = cls_conv_out.shape
        c_plus_1 = int(c_plus_1_times_kk/(k**2))
        pooling_track = torch.zeros((n,c_plus_1,k,k))
        scores = torch.zeros((n,c_plus_1))
    
        for i in range(0,n):
            ymin = int(rois[i,0])
            xmin = int(rois[i,1])
            ymax = int(rois[i,2])
            xmax = int(rois[i,3])
        
            y_range = ymax - ymin
            x_range = xmax - xmin
            y_step = int(y_range/k)
            x_step = int(x_range/k)
        
            count = 0
            for j in range(0,k):
                y_start = ymin+j*y_step
                y_stop = y_start+y_step
                for l in range(0,k):
                    x_start = xmin+l*x_step
                    x_stop = x_start+x_step
                    #print(y_start,y_stop,x_start,x_stop)
                    for cls in range(0,c_plus_1):
                        pooling_track[i,cls,j,l] = cls_conv_out[_-1,count,y_start:y_stop,x_start:x_stop].mean()
                        count=count+1
        
        pooling_track[pooling_track != pooling_track] = 0   # this will convert all of th nan value to zero
        scores = self.softmax(self.AvgPool(pooling_track.float()))
        return scores


# In[2]:


#PSSM =PositionSensitiveScoreMap()
#print(PSSM)
#scores = PSSM(cls_conv_out,rois)


# In[ ]:




