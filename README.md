
# Multi-Object-Detection 

## Description
Our goal was to modify it into an R-FCN. This project is based off the faster rcnn https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/README.MD

Uses PascalVOC detection dataset for multi-object detection in PyTorch.

#### This repo is a work in progress - we did not finish the entire implementation (end to end) without bugs

## Requirements

#### Install the following to your python or anaconda environment:
torch,
visdom,
torchvision,
scikit-image,
tqdm,
fire,
pprint,
matplotlib,
ipdb,
cython,
cupy,
git+https://github.com/pytorch/tnt.git@master,

#### example install of packages
Install package 'visdom' as follows:    
       pip install --user visdom
      


## Prerequisites

You will need to get the Pascal VOC 2012 dataset from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar


### Installing

requires PyTorch >=0.4

-install PyTorch >=0.4 with GPU (code are GPU-only), refer to official website

-install cupy, you can install via pip install cupy-cuda80 or(cupy-cuda90,cupy-cuda91, etc).

-install other dependencies: pip install -r requirements.txt

-Optional, but strongly recommended: build cython code nms_gpu_post:

    cd model/utils/nms/
    python build.py build_ext --inplace
    cd -

-start visdom for visualization

nohup python -m visdom.server &


## Code organization

demo.ipynb - Load a trained model and run a demo

train.ipynb - Run a training of the model

train.py - Function and wrapper used for training

trainer.py - FasterRCNN wrapper used to iterate through train steps for R-FCN 

model/r_fcn - R-FCN base class to be inherited and have the extractor, RPN, and head defined
    
model/region_proposal_network - region proposal network from Faster-RCNN
    
model/resnet101extractor - feature extractor class using resnet101 with custom convolution dimensionality reduction layer
    
model/rfcn_resnet101 - RFCN model based on Resnet101 with head class for PSROIPooling, etc
    
model/roi_module - Region of interest module
    
model/utils/bbox_tools - process and create bounding boxes
        
model/utils/creator_tool - generate anchor targets, proposal targets, and proposal regions

model/utils/roi_cupy - unused RoI generator for Faster-RCNN
    
    
    nms/
        _nms_gpu_post.c - Faster RCNN non maximum suppression in C
        _nmu_gpu_post.pyx
        _nms_gpu_post.cpython-36m-x86_64-linux-gnu.so
        _nms_gpu_post_py.py - Faster RCNN cms code
        build.py - Build cython code for non maximum suppression
        non_maximum_suppression.py - Perform non maximum suppression by IoUs
utils/

PositionSensitiveScoreMap_V2.py           --  Implement PS-ROI pooling

PositionSensitiveScoreMap_deprecated.py   --  Implement PS-ROI pooling (old version, don't use)

ProcessScores.py                          --  Threshold ROI scores

SisterRegressionROIPooling.py             --  PS-ROI pooling for bbox regression

Sister_4_k_squared_conv.py                --  Convolution for bbox regression

__init__.py                               --  Ignore

array_tool.py                             --  Tools to convert arrays

bbox_roi_to_paramaterized_t.py            --  Change ROI format

config.py                                 --  Constant definitions

dataloader.py                             --  Dataset loader and tools

eval_tool.py                              --  Evaluation tools

iou_est.py                                --  Calculate IOU score

vistool.py                                --  Tools to plot images and losses




## Authors
This work was modified for a project by:
Felix Fagan
Mike Ranis
Pryor Vo






```python

```
