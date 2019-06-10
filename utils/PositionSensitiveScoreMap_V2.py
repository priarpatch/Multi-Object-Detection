'''
Raw implementation of position sensitive score mapping for classification
'''

import torch
import torch.nn as nn
import torch.nn.functional as F




class PositionSensitiveScoreMap_V2(nn.Module):
    
    # inputs to the forward pass ar the classification convolution layer outputs
    # and rois
    # Outputs: scores for all C+1 and for each bbox in the rois 
    
    # Here we define our network structure
    def __init__(self,feat_stride = (520//16),k=3):
        super(PositionSensitiveScoreMap_V2, self).__init__()
        self.feat_stride = feat_stride
        self.k=k
        self.softmax = nn.Softmax()
        
    # Here we define one forward pass through the network
    def forward(self, cls_conv_out,rois):
        '''
        cls_conv_out: tensor(N,K^2*(C+1),H,W)
        rois: tensor(N,4)
        '''
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
            ymax = int(rois[i,2])+1
            xmax = int(rois[i,3])+1
        
            y_range = ymax - ymin
            x_range = xmax - xmin
            y_step = int(y_range/k)
            x_step = int(x_range/k)
            
            block = F.adaptive_avg_pool2d(cls_conv_out[:,:,ymin:ymax,xmin:xmax],(7,7))
         
            count = 0
            for j in range(0,k):
                for l in range(0,k):
                    for cls in range(0,c_plus_1):
                        pooling_track[i,cls,j,l]=block[:,count,j,l]
                        #old pooling track version was here
                        count = count +1
        
        pooling_track[pooling_track != pooling_track] = 0   # this will convert all of th nan value to zero
        scores = self.softmax(F.adaptive_avg_pool2d(pooling_track.float(),(1,1))[:,:])
        scores = torch.squeeze(scores)
        
        return scores



