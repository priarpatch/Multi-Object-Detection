#!/usr/bin/env python
# coding: utf-8

# In[48]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class SisterRegressionROIPooling(nn.Module):
    
    # inputs to the feature_map and rois
    # does the convolution to get a 4*(k^2)-d output and then does roi sensitive pooling described as Bounding Box Regression
    # Outputs: a tensor that has n 4-d tensors, where n is the number of rois input into the forward path 
    
    # Here we define our network structure
    def __init__(self,feat_stride=520//16,x_scale = 500//16,y_scale = 330//11 ,k=3):
        super(SisterRegressionROIPooling, self).__init__()
        #self.x_scale = x_scale
        #self.y_scale = y_scale
        self.feat_stride = feat_stride
        self.k=k

    # Here we define one forward pass through the network
    def forward(self,conv_out,rois):
        #x_scale = self.x_scale
        #y_scale = self.y_scale
        feat_stride = self.feat_stride
        rois = rois // feat_stride

        k=self.k
        
        
        n = rois.shape[0] #number of regions to look at
        
        out = torch.zeros((n,2,4))
    
        for i in range(0,n):
            ymin = int(rois[i,0])
            xmin = int(rois[i,1])
            ymax = int(rois[i,2])+1
            xmax = int(rois[i,3])+1
            
            
            #y_range = ymax - ymin # height range
            #x_range = xmax - xmin # width range
            #y_step = int(y_range/k)
            #x_step = int(x_range/k)
            
           
            
            #print(ymin,ymax,xmin,xmax)
            
            #condition_1 = ((ymax-ymin)<1)
            
            
            ##############################################################
            block = conv_out[:,:,ymin:ymax,xmin:xmax] # takes th ROI block
            #block[block != block] = 0 #removes NaNs
            block = F.adaptive_avg_pool2d(block, (7,7))
            block = F.adaptive_avg_pool2d(block, (1,1))
            #############################################################
            count = 0
            for cls in range(0,2):
                for z in range(0,4):
                    out[i,cls,z] = block[0,count] #roi,cls,coordinate indexing
                    count=count+1
        #out is a nx20x4 tensor
        return out





