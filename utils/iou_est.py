import numpy as np
from copy import deepcopy

def iou_est(bbox_truth, bbox_test):
    '''
    Estimates Intersection over Union (IoU) for object detection.
    Inputs:
        box_truth: tuple containing bounding box coordinates of ground-truth image region. Tuple is in ths form [upper left X, upper
                   left Y, lower right X, lower right Y]
        box_test:  tuple containing bounding box coordinates of proposed image region from network output. Tuple is in ths form [upper
                   left X, upper left Y, lower right X, lower right Y]
        
    Outputs: (image, mask, objects)
        iou: IoU score of proposed image region. Equal to the ration of overlap area of ground-truth and proposed bounding boxes and
             the area of union of the two regions.
                
    '''
    from copy import deepcopy
    
    box_truth = deepcopy(bbox_truth)
    box_test  = deepcopy(bbox_test)
    
    # Get coordinates of bounding boxes
    xmin_truth = box_truth[0] # minimum x-coordinate of ground-truth box
    ymin_truth = box_truth[1] # minimum y-coordinate of ground-truth box
    xmax_truth = box_truth[2] # maximum x-coordinate of ground-truth box
    ymax_truth = box_truth[3] # maximum y-coordinate of ground-truth box
    
    xmin_test  = box_test[0]  # minimum x-coordinate of proposed box
    ymin_test  = box_test[1]  # minimum y-coordinate of proposed box
    xmax_test  = box_test[2]  # maximum x-coordinate of proposed box
    ymax_test  = box_test[3]  # maximum y-coordinate of proposed box
    
    # calculate areas of individual bounding boxes
    area_truth = (xmax_truth - xmin_truth) * (ymax_truth - ymin_truth)
    area_test  = (xmax_test - xmin_test) * (ymax_test - ymin_test)
    
    # check overlap condition - if true calculate overlap area, set to zero otherwise
    x_overlap = ((xmin_truth <= xmin_test <= xmax_truth) or (ymin_truth <= xmax_test <= ymax_truth))
    y_overlap = ((ymin_truth <= ymin_test <= ymax_truth) or (ymin_truth <= ymax_test <= ymax_truth))
    
    if x_overlap and y_overlap:
        xmin_ol = max(xmin_truth, xmin_test) # minimum x-coordinate of overlap box
        ymin_ol = max(ymin_truth, ymin_test) # minimum y-coordinate of overlap box
        xmax_ol = min(xmax_truth, xmax_test) # maximum x-coordinate of overlap box
        ymax_ol = min(ymax_truth, ymax_test) # maximum y-coordinate of overlap box
        
        area_ol = (xmax_ol - xmin_ol) * (ymax_ol - ymin_ol)
    else:
        area_ol = 0
    
    # calculate union area and IoU
    area_union = area_truth + area_test - area_ol
    iou        = area_ol / area_union
    
    return iou