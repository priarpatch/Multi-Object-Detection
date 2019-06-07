import torch as t
import torch.nn as nn
import torch.utils.data as td
import torch.nn.functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from utils import array_tool as at

# might need this later
# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

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
        roi_cls_loss, roi_loc_loss = self.roi_loss(roi_loc, roi_cls_loc, gt_roi_loc, gt_roi_lbl)
        
        
        
        
        
        
        total = rpn_loc_loss + rpm_cls_loss + roi_loc_loss + roi_cls_loss
        
        # not sure if losses should be a dictionary instead, but here's a definition for that just in case
        #losses = {
        #    'rpn_loc_loss': rpn_loc_loss,
        #    'rpn_cls_loss': rpn_cls_loss,
        #    'roi_loc_loss': roi_loc_loss,
        #    'roi_cls_loss': roi_cls_loss,
        #    'total_loss'  : total
        #}
        losses = [rpn_loc_loss, rpm_cls_loss, roi_loc_loss, roi_cls_loss, total]
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
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_lbl.to(self.device), ignore_index=-1)
        _gt_rpn_lbl = gt_rpn_lbl[gt_rpn_lbl > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_lbl) > -1]
        #self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_lbl.data.long())
        
        
        
        
        return rpn_loss_cls, rpn_loss_loc
    
    
    def roi_loss(self, loi_loc, roi_cls_loc, gt_roi_loc, gt_roi_lbl):
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().to(self.device), \
                              at.totensor(gt_roi_lbl).long()]
        gt_roi_lbl = at.totensor(gt_roi_lbl).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = self.loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_lbl.data, 1)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_lbl.self.device)
        
        return roi_cls_loss, roi_loc_loss
    
    
    def loc_loss(self, loc_pred, loc_gt, lbl_gt, sigma):
        in_weight = t.zeros(loc_gt.shape).to(self.device)
        
        in_weight[(lbl_gt > 0).view(-1,1).expand_as(in_weight).to(self.device)] = 1
        
        loc_loss = self.smooth_L1_loss(loc_pred, loc_gt, in_weight.detach(), sigma)
        
        # normalize loss by total num ber of positive and negative rois
        loc_loss = loc_loss / ((lbl_gt >= 0).sum().float()) #ignore lbl_gt==-1 for rpn loss
        
        return loc_loss
    
    
    def smooth_L1_loss(self, x1, x2, in_weight, sigma):
        # Calculate localization loss using Smooth L1 loss, as defined in R. Girshick. Fast R-CNN. InICCV, 2015
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

'''        
def smooth_L1(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws):
    # Smooth L1 loss, as defined in R. Girshick. Fast R-CNN. InICCV, 2015
    ## UNDER CONSTRUCTION
    return loss

def RCNN_loss(cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws):
    # note: cls_score and rois_label are softmax score vectors, NOT one-hot vectors
    # classification loss (cross entropy)
    class_loss = F.cross_entropy(cls_score, rois_label)
    
    # bounding box regression loss (smooth L1 loss)
    bbox_loss = smooth_L1(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
    
    return class_loss, bbox_loss

def rpn_loss(rpn_class_score, rpn_label):
    class_loss = F.cross_entropy(rpn_cls_score, rpn_label)
    
    # this one not done, gotta edit arguments
    bbox_loss = smooth_L1(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
    
    return class_loss, bbox_loss

def train(model, train_set, B=1, lr=1e-3, device, num_epoch):
    params = []
    for name, value in dicr(model.named_parameters()).items():
        if value.requires_grad():
            param = param + [{'params:':[name],'lr':lr,'weight-decay':0.0005}]
        optimizer =torch.optim.Adam(params)
        
        # load checkpoint here
        model.to(device)
        
        for epoch in range(num_epoch):
            model.train()
            
            # adjust learning rate every x epochs here if needed
            
            for bat_ind, (image, bbox, bbox_labels) in enumerate(train_loader):
                [_,h,w] = image.size()
                image = image.view(1,3,h,w)
                
                model.zero_grad()
                
                # forward pass
                rois, class_prob, bbox_pred, rois_labl = model(image)
                # RCNN loss
                RCNN_loss_cls, RCNN_loss_bbox = RCNN_loss(cls_score, rois_label, bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
                
                # rpn loss
                rpn_loss_cls, rpn_loss_box = rpn_loss()#loss function for rpn
                
                #total loss
                loss = rpn_loss_cls + rpn_loss_box + RCNN_loss_cls + RCNN_loss_cls
                
                #back prop
                optimizer = zero_grad()
                loss.backward()
        
        #
'''