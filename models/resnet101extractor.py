#import os
#import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.utils.data as td
import torchvision as tv

# to use this code for forward propagation, do the following:
# $import models.resnet101extractor as rn
# $net = rn.Resnet101Extractor(20)          # Instantiates net (can leave out the 20 as num_classes=20 by default)
# $net = net.to(device)                    # Put net on device of your choice (GPU, etc.) 
# $y = net(x.view(1,3,H,W))                # Forward pass. Be sure to resize RGB image from (3,H,W) tensor to (1,3,H,W) beforehand


class Resnet101Extractor(nn.Module):
    '''
    This class defines the portion of the architechture of R-FCN that uses ResNet101. 
    Inputs:
        num_classes: number of classes for classifier. Default is 20
        
    Attributes:
    Following are default layers from Resnet101
    conv1  :
    bn1    :
    maxpool:
    layer1 :
    layer2 :
    layer3 :
    layer4 : --> last convolutional block has 2048-d output
    Following layers have been added
    conv_custom  : 1024-d convolutional layer - reduces dimensions
                
    '''
    def __init__(self, num_classes=20, fine_tuning=False):
        
        super(Resnet101Extractor, self).__init__()
        resnet = tv.models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = fine_tuning
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # avgpool and fc from Resnet101 are ignored
        # replace with single randomly initialized conv layer
        # initialization is from normal distribution - subject to change
        self.conv_custom = nn.Conv2d(2048, 1024, 1)
        self.conv_custom.weight.data = nn.init.normal_(self.conv_custom.weight.data, mean=0, std=1)
    
    def forward(self, x):
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        y = self.conv_custom(f)
        
        return y