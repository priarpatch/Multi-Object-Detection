#!/usr/bin/env python
# coding: utf-8

# In[48]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from model.roi_module import RoIPooling2D


class SisterRegressionROIPooling(nn.Module):
    
    # inputs to the feature_map and rois
    # does the convolution to get a 4*(k^2)-d output and then does roi sensitive pooling described as Bounding Box Regression
    # Outputs: a tensor that has n 4-d tensors, where n is the number of rois input into the forward path 
    
    # Here we define our network structure
    def __init__(self,feat_stride=520//16,x_scale = 500//16,y_scale = 330//11 ,k=7):
        super(SisterRegressionROIPooling, self).__init__()
        #self.x_scale = x_scale
        #self.y_scale = y_scale
        #self.feat_stride = feat_stride
        #self.k=k
        self.ROI_1 = RoIPooling2D(3,3,1/16).cuda()

    # Here we define one forward pass through the network
    def forward(self,conv_out,rois):
        #x_scale = self.x_scale
        #y_scale = self.y_scale
        #feat_stride = self.feat_stride
        #rois = rois // feat_stride

        #k=self.k
        
        
        print('conv_out shape for sister roi pool is')
        print(conv_out.size())
        
        
        n = rois.shape[0] #number of regions to look at
        #sample_roi = torch.zeros(n,4).cuda()
        #sample_roi[:,2:4]=3
        
        with torch.enable_grad():
            ROI_Pooling_Out_1 = self.ROI_1(conv_out,rois).cuda()
            ROI_Pooling_Out_1 = F.adaptive_avg_pool2d(ROI_Pooling_Out_1,(1,1)).squeeze().cuda()
            ROI_Pooling_Out_1 = ROI_Pooling_Out_1.view(n,-1,4)
    
        
        return ROI_Pooling_Out_1





