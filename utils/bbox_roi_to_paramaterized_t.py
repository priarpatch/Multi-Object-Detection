#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

def bbox_roi_to_paramaterized_t(roi):
    # assuming the roi is [ymin,xmin,ymax,xmax]
    tx = roi[1] #xmin
    tw = roi[4] - tx #xmax - xmin
    ty = roi[0] #ymin
    th = roi[3] - ty #ymax - ymin
    t= torch.tensor([tx,ty,tw,th])
    return t
    

