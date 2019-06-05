import torch.utils.data as td
import torch.nn.functional as F

'''
def eval(dataloader, model):
    box_pred    = []
    label_pred  = []
    box_truth   = []
    label_truth = []
    
    result = #evaluate function
    
    return result

def train(model, train_set, device = 'cuda', B=1, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    adam = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = td.DataLoader(train_set, batch_size = B, pin_memory = True, shuffle = True)
    
    best_map = 0
    
    for epoch in range():
        #clear stuff
        for bch_in, (image, bbox, bbox_labels) in enumerate(train_loader):
            img  = image.to(device)
            bbox = bbox.to(device)
            lbl  = bbox_labels.to(device)
            
            #trainer.step(img, bbox, label)
            adam.zero_grad()
            losses = forward(img, bbox, lbl)
            losses.total_loss.backward()
            adam.step()
            
            
            #plot shit
    
        result = (train_loader, model)
        #plot
        
        
        if result['map'] > best_map:
            best_map = result['map']
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
