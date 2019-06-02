import os
import numpy as np
import torch
import torch.utils.data as td
import torchvision as tv
from PIL import Image
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('ymin').text),
                              int(bbox.find('xmin').text),
                              int(bbox.find('ymax').text),
                              int(bbox.find('xmax').text)]
        objects.append(obj_struct)

    return objects


class PascalVOCDataset(td.Dataset):
    '''
    objects: list of the annotation dictionaries with keys {name, pose, truncated, difficult, bbox}. 
                name and bbox are probably the more important ones, which are already used in the mask.
    Inputs:
        root_dir: Directory of PascalVOC2012 dataset
        mode: "train", "val", "trainval"
        
        
    Attributes:
    class_dict : dictionary describing how different classes are encoded
    __len__() : returns length of dataset
    __getitem__(idx) 
        Outputs: (image, bbox, bbox_labels, image_index)
            image: the image as a torch tensor of values between [-1,1] of dimensions (3,h,w)
            bbox: masks of bounding box labels, torch tensor of dimensions(K,h,w) where K is number of classes
            bbox_labels: object labels of boxes
            
                
    '''
    def __init__(self, root_dir, mode = 'train', im_size = 520):
        super(PascalVOCDataset, self).__init__()
        self.files = {}
        self.mode = mode
        self.im_size = im_size
        
        self.class_dict = {"person":0, "bird":1, "cat":2, "cow":3,
                           "dog":4, "horse":5, "sheep":6,
                           "aeroplane":7, "bicycle":8, "boat":9, "bus":10, "car":11,
                           "motorbike":12, "train":13,"bottle":14, "chair":15, 
                           "diningtable":16, "pottedplant":17, "sofa":18, "tvmonitor":19}
        
        self.num_classes = len(self.class_dict.keys())
        
        for split in ["train", "val", "trainval"]:
            path = os.path.join(root_dir, "ImageSets/Main", split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
            
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        self.annot_dir = os.path.join(root_dir, "Annotations")
    
    def __len__(self):
        return len(self.files[self.mode])
    
    def __repr__(self):
        return "PascalVOCDataset(mode={})". \
            format(self.mode)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, \
        self.files[self.mode][idx]+'.jpg')
        
        annot_path = os.path.join(self.annot_dir, \
        self.files[self.mode][idx]+'.xml')
        
        objects = parse_rec(annot_path)
        
        img = Image.open(img_path).convert('RGB')
        
        transform = tv.transforms.Compose([
        tv.transforms.Resize((self.im_size,self.im_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        
        image = transform(img)
        bbox_labels = torch.zeros((len(objects),self.num_classes))
        i=0
        bbox = list()
        bbox_imidx = list()
        for obj in objects:
            obj_idx = self.class_dict[obj['name']]
            bbox_labels[i,obj_idx]=1
            bbox.append(obj['bbox'])
            bbox_imidx.append([idx])
            i+=1
        bbox = torch.tensor(bbox)
        bbox_imidx = torch.tensor(bbox_imidx)
        return image, bbox, bbox_labels#, bbox_imidx
    
def myimshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h

