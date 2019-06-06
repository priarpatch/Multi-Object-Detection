#!/usr/bin/env python
# coding: utf-8




import torch

class ProcessScores():
    
    #takes the scores ouput and processes it with thresholding and only returns rois above .6 scores

    def indices(self,a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    def process_scores_with_thresholding(self,scores,rois):
        #process_scores_with_thresholding_and_return_classes_with_valid_rois
        # assume n is = rois.shape[0] , number of rois
        # cls options are C+1 , so indices will be on 0 to 20 , with 20 being background
        cls_list = list()
        max_scores_list = list()
        max_scores = scores.max(1)[0] # get only the largest score from each of the rois across the 21 class options
        valid_roi_indices_from_thresholding = self.indices(max_scores, lambda x: x >= 0.6) # find the indices of all max_scores >=0.6
    
        num_valid = len(valid_roi_indices_from_thresholding) # this is the number of rois that passed the score thresholding
        valid_rois = torch.zeros((num_valid,4)) #this is for outputting just the valid rois
    
        for i in range(0,num_valid):
            roi_index = valid_roi_indices_from_thresholding[i] # this is also the score index, this is the roi index that created the score 
            cls_of_roi = self.indices(scores[roi_index], lambda x: x == scores[roi_index].max()) # gets class of the valid max_score (this is a list)
            cls_list.append(cls_of_roi[0]) # this appends the integer of the class of the roi to the cls_list
            valid_rois[i,:] = torch.Tensor(rois[roi_index,:])
            max_scores_list.append(float(scores[roi_index].max())) # in case we need to choose the highest scores or threshold further
    
        return cls_list,valid_rois,max_scores_list






