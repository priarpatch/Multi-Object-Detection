from __future__ import  absolute_import
import torch as t
from torch import nn
from model.resnet101extractor import Resnet101Extractor
from model.region_proposal_network import RegionProposalNetwork
from model.r_fcn import RFCN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt
from utils.PositionSensitiveScoreMap_V2 import PositionSensitiveScoreMap_V2 as PositionSensitiveScoreMap
from utils.SisterRegressionROIPooling import SisterRegressionROIPooling

class RFCNResnet101(RFCN):
    """R-FCN based on Resnet-101.
    For descriptions on the interface of this model, please refer to
    :class:`model.r_fcn.RFCN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 520//16  # Res101 downsamples from ~520 to ~17 #changed this

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor = Resnet101Extractor() #changed this
        
        rpn = RegionProposalNetwork(
            1024, 1024, #changed this. Resnet101 with custom layer results in 1024 channels
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = Resnet101RoIHead( #Main class to change
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
        )

        super(RFCNResnet101, self).__init__(
            extractor,
            rpn,
            head,
        )


class Resnet101RoIHead(nn.Module):
    """R-FCN head for Resnet101 based implementation.
    This class is used as a head for R-FCN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes including background (C+1)
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale, k=3):
        # n_class includes background
        super(Resnet101RoIHead, self).__init__()

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        
        #k^2(C+1) convolution for classification
        self.cls_layer = nn.Conv2d(1024,self.n_class*k*k, [1,1], padding=0, stride=1)
        nn.init.normal(self.cls_layer.weight.data, 0.0, 0.01)
        
        #4k^2*(C) convolution for regression
        self.reg_layer = nn.Conv2d(1024, 4*2*k*k, [1,1], padding=0, stride=1)
        nn.init.normal(self.reg_layer.weight.data, 0.0, 0.01)
        
        #Position-Sensitive ROI Pooling, voting
        self.PSROI_cls = PositionSensitiveScoreMap(k=k)
        self.PSROI_reg = SisterRegressionROIPooling(k=k)
        
#         #Avg pooling (voting)
#         self.cls_score = nn.AvgPool2d((7,7), stride=(7,7))
#         self.bbox_pred = nn.AvgPool2d((7,7), stride=(7,7))
        
        ##not sure if this layer will continue to work with new architecture##
        #self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)
        

    def forward(self, feature_maps, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            feature_maps (Variable): 4D feature map representation of images variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        
        # in case roi_indices is  ndarray
        #roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        #indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        
        #xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        #indices_and_rois =  xy_indices_and_rois.contiguous()
        
        cls_out = self.cls_layer(feature_maps)
        roi_scores = self.PSROI_cls(cls_out, rois)
#         cls_scores = self.cls_score(cls_out)
        reg_out = self.reg_layer(feature_maps)
        roi_cls_locs = self.PSROI_reg(reg_out, rois)
#         bbox_preds = self.bbox_pred(bbox_reg)
#         pool = self.roi(x, indices_and_rois)
#         pool = pool.view(pool.size(0), -1)
#         fc7 = self.classifier(pool)
#         roi_cls_locs = self.cls_loc(fc7)
#         roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
