#!/usr/bin/env python
# coding: utf-8

# In[48]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class Sister_4_k_squared_conv(nn.Module):
    
    # inputs to the forward pass ar the classification convolution layer outputs
    # and rois
    # Outputs: scores for all C+1 and for each bbox in the rois 
    
    # Here we define our network structure
    def __init__(self,feat_stride = 520//17,k=7):
        super(Sister_4_k_squared_conv, self).__init__()
        self.feat_stride = feat_stride
        self.k=k
        self.conv1 = nn.Conv2d(1024,4*k*k,(1,1))

    # Here we define one forward pass through the network
    def forward(self, feature_map,rois):
        feat_stride = self.feat_stride
        rois = rois // feat_stride
        k=self.k
        
        n = rois.shape[0] #number of regions to look at
        
        conv_out = self.conv1(feature_map)
        
        pooling_track = torch.zeros((n,4,k,k))
        #scores = torch.zeros((n,4))
        
        n = rois.shape[0] #number of regions to look at
    
        t = torch.zeros((n,4))
    
        for i in range(0,n):
            ymin = int(rois[i,0])
            xmin = int(rois[i,1])
            ymax = int(rois[i,2])
            xmax = int(rois[i,3])
            
            
            y_range = ymax - ymin # height range
            x_range = xmax - xmin # width range
            y_step = int(y_range/k)
            x_step = int(x_range/k)
            
            ##############################################################
            ## not used yet ## 
            tlx = xmin # top_left x
            tly = ymin # top_left y
            center_x = (xmin+xmax) / 2
            center_y = (ymin+ymax) / 2 
            tx = center_x  
            ty = center_y
            tw = x_range 
            th = y_range
            
            #############################################################
            count = 0
            for j in range(0,k):
                y_start = ymin+j*y_step
                y_stop = y_start+y_step
                for l in range(0,k):
                    x_start = xmin+l*x_step
                    x_stop = x_start+x_step
                    #print(y_start,y_stop,x_start,x_stop)
                    for z in range(0,4):
                        im = conv_out[:,count,y_start:y_stop,x_start:x_stop]
                        pooling_track[i,z,j,l] = im.mean()
                        count=count+1
                        
        
        pooling_track[pooling_track != pooling_track] = 0   # this will convert all of th nan value to zero
        out = F.adaptive_avg_pool2d(pooling_track, (1,1))
        
        return out


# In[ ]:




