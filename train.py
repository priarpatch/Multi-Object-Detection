from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import matplotlib
from tqdm import tqdm

import torch as t
from utils.config import opt
from model.rfcn_resnet101 import RFCNResnet101
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import torch.nn as nn
import torch.utils.data as td
import torch.nn.functional as F

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    gt_difficults = False
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs)
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(train_set, val_set, load_path = False, epochs = 1, lr=1e-3, record_every = 300, lr_decay = 1e-3,test_num=500):
    '''
    Uses the training set and validation set as arguments to create dataloader. Loads and trains model
    '''
    train_dataloader = td.DataLoader(train_set, batch_size = 1, pin_memory = False, shuffle = True)
    test_dataloader = td.DataLoader(val_set, batch_size = 1, pin_memory = True)
    faster_rcnn = RFCNResnet101().cuda()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    saved_loss = []
    iterations = []
    if load_path:
        trainer.load(load_path)
        print('load pretrained model from %s' % load_path)
        state_dict = t.load(load_path)
        saved_loss = state_dict['losses']
        iterations = state_dict['iterations']
        
    best_map = 0
    lr_ = lr
    for epoch in range(epochs):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(train_dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            losses = trainer.train_step(img, bbox, label, scale)
            loss_info = 'Iter {}; Losses: RPN loc {}, RPN cls: {}, ROI loc {}, ROI cls {}, Total:{}'.format(
                                                str(ii),
                                                "%.3f" % losses[0].cpu().data.numpy(),
                                                "%.3f" % losses[1].cpu().data.numpy(),
                                                "%.3f" % losses[2].cpu().data.numpy(),
                                                "%.3f" % losses[3].cpu().data.numpy(),                                
                                                "%.3f" % losses[4].cpu().data.numpy())
            print(loss_info)
            if (ii + 1) % record_every == 0:
                
                iterations.append(ii + 1) 
                saved_loss.append([losses[0].cpu().item(),losses[1].cpu().item(),
                              losses[2].cpu().item(),losses[3].cpu().item(),
                              losses[4].cpu().item()])
                kwargs = {"losses": saved_loss, "iterations": iterations}
                trainer.save(saved_loss = saved_loss, iterations = iterations)
                print("new model saved")
                
class RFCNtrainer(nn.Module):
    def __init__(self, model, optimizer, device):
        super(RFCNtrainer, self).__init__()
        
        self.model     = model
        self.optimizer = optimizer
        self.device    = device
        
        # Put model and optimizer on device
        self.model     = self.model.to(self.device)
        self.optimizer = self.model.to(self.device)
        
        self.anchor_target_creator   = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
    
    def eval(dataloader, model):
        box_pred    = []
        label_pred  = []
        box_truth   = []
        label_truth = []

        result = 0 #evaluate function

        return result
    
    def train(self, train_set, test_set, num_epoch, B=1, lr=1e-3):
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        #model = model.to(device)
        #adam = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_loader = td.DataLoader(train_set, batch_size=B, pin_memory = False, shuffle = True)
        test_loader  = td.DataLoader(test_set, batch_size=B, pin_memory = True, shuffle = False)
        
        # load stuff here from log file
        
        best_map = 0
        
        self.model.zero_grad()
        
        # set up plots here
        
        for epoch in range(num_epoch):
            #clear stuff (RFCNtrainer.reset_meters())
            for batch_ind, (image, bbox, bbox_labels, scale) in enumerate(train_loader):
                #move data to device
                scale = at.scalar(scale)
                img  = image.to(self.device)
                bbox = bbox.to(self.device)
                lbl  = bbox_labels.to(self.device)
                self.step(img, bbox, lbl, scale)
                
            #plot loss and stuff every 2 epochs
            if (epoch+1) % 2 == 0:
                # plot stuff (loss, boxes, rpn confusion matrix, etc.) goes here
                emptyval = []
            
        # test with evaluation data, plot results #-->
        #result = eval(train_loader, self.model) #-->
        
        # log info to file here #-->
        
        #plot #-->
        
        
        #if (result['map'] > best_map): #-->
        #    best_map = result['map']#-->
        
        return
        
    def step(self, imgs, bboxes, lbls, scale):
        #forward pass through network, get losses, then backprop
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, lbls, scale)
        losses[-1].backward()
        self.optimizer.step()

        return losses
    
    def forward(self, imgs, bboxes, lbls, scale):
        # forward pass, get losses as tuple
        n = bboxes.shape[0]
        _,_,H,W = imgs.shape #H, W = dimensions of images
        im_size = (H,W)
        
        features = self.model.extractor(imgs)
        
        rpn_locs, rpn_scores, rois, roi_ind, anchor = self.model.rpn(features, im_size, scale)
        
        #batch size = 1, therefore make variable singular
        rpn_loc = rpn_locs[0]
        rpn_score = rpn_scores[0]
        roi = rois
        bbox  = bboxes[0]
        lbl = lbls[0]
        
        sample_roi, gt_roi_loc, gt_roi_lbl = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(lbl),
            self.model.loc_normalize_mean,
            self.model.loc_normalize_std
        )
        sample_roi_ind = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.model.head(features, sample_roi, sample_roi_ind)
        
        
        
        
        
        # ----- RPN Losses -----
        rpn_cls_loss, rpn_loc_loss = self.rpn_loss(rpn_loc, rpn_score, bbox, anchor, im_size)
        
        # ----- ROI losses -----
        roi_cls_loss, roi_loc_loss = self.roi_loss(roi_cls_loc, gt_roi_loc, gt_roi_lbl)
        
        
        
        
        
        
        total = rpn_loc_loss + rpm_cls_loss + roi_loc_loss + roi_cls_loss
        
        # not sure if losses should be a dictionary instead, but here's a definition for that just in case
        #losses = {
        #    'rpn_loc_loss': rpn_loc_loss,
        #    'rpn_cls_loss': rpn_cls_loss,
        #    'roi_loc_loss': roi_loc_loss,
        #    'roi_cls_loss': roi_cls_loss,
        #    'total_loss'  : total
        #}
        losses = [rpn_loc_loss.to(self.device), rpm_cls_loss.to(self.device), roi_loc_loss.to(self.device), roi_cls_loss.to(self.device), total.to(self.device)]
        return losses
    
    
    def rpn_loss(self, rpn_loc, rpn_score, bbox, anchor, im_size):
        ## Get ground truth locs and labels
        #gt_rpn_loc, gt_rpn_lbl = self.anchor_target_creator(at.tonumpy(bbox), \
        #                                                          anchor, im_size)
        #gt_rpn_lbl = at.totensor(gt_rpn_loc).long()
        #gt_rpn_loc = at.totensor(gt_prn_loc)
        #
        ## calculate localization loss for rpn (sigma = 3)
        #rpn_loss_loc = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_lbl.data, 3)
        #
        ## calculate classification loss for rpn
        #rpn_loss_cls = F.cross-entropy(rpn_score, gt_prn_lbl.device(), ignore_index=-1)
        
        ## Get ground truth locs and labels
        gt_rpn_loc, gt_rpn_lbl = self.anchor_target_creator(at.tonumpy(bbox), anchor, im_size)
        gt_rpn_lbl = at.totensor(gt_rpn_lbl).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        
        ## calculate localization loss for rpn (sigma = 3)
        rpn_loc_loss = self.loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_lbl.data, 3)

        # calculate classification loss for rpn
        rpn_cls_loss = F.cross_entropy(rpn_score.to(self.device), gt_rpn_lbl.to(self.device), ignore_index=-1)
        _gt_rpn_lbl = gt_rpn_lbl[gt_rpn_lbl > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_lbl) > -1]
        #self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_lbl.data.long())
        
        
        
        
        return rpn_cls_loss, rpn_loc_loss
    
    
    def roi_loss(self, roi_cls_loc, gt_roi_loc, gt_roi_lbl):
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        
        #print('arg 1:',t.arange(0, n_sample).long().shape)
        #print('arg 2:',at.totensor(gt_roi_lbl).long().shape)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().to(self.device), \
                              at.totensor(gt_roi_lbl).long()]
        gt_roi_lbl = at.totensor(gt_roi_lbl).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = self.loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_lbl.data, 1)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score.to(self.device), gt_roi_lbl.to(self.device))
        
        return roi_cls_loss, roi_loc_loss
    
    
    def loc_loss(self, loc_pred, loc_gt, lbl_gt, sigma):
        in_weight = t.zeros(loc_gt.shape).to(self.device)
        
        in_weight[(lbl_gt > 0).view(-1,1).expand_as(in_weight).to(self.device)] = 1
        
        loc_loss = self.smooth_L1_loss(loc_pred, loc_gt, in_weight, sigma)
        
        # normalize loss by total num ber of positive and negative rois
        loc_loss = loc_loss / ((lbl_gt >= 0).sum().float()) #ignore lbl_gt==-1 for rpn loss
        
        return loc_loss
    
    
    def smooth_L1_loss(self, x1, x2, in_weight, sigma):
        # Calculate localization loss using Smooth L1 loss, as defined in R. Girshick. Fast R-CNN. InICCV, 2015
        #in_weight = in_weight.to(t.double)
        in_weight = in_weight.float()
        x1 = x1.float()
        x2 = x2.float()
        
        sigma2 = sigma**2
        arg = in_weight * (x1 - x2)
        abs_arg = arg.abs()
        arg_less_than_1 = (abs_arg.data < (1. / sigma2)).float() # change boolean to float, multiply difference with this to filter out values from specific cases, as seen below
        #case 1: abs(arg) < 1
        y_case1 = (arg_less_than_1) * (0.5 * sigma2) * (arg**2)
        #case 2: abs(arg) >= 1
        y_case2 = (1 - arg_less_than_1) * (abs_arg - (0.5 /  sigma2))
        
        y = y_case1 + y_case2
        
        #x_less_than 1 = (abs_diff.data < (1. / sigma2)).float()
        #y = (flag * (sigma2 / 2.) * (diff**2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
        #loc_loss = y.sum()
        
        return y

