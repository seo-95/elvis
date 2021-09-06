import json
import os
import pdb
import random

#import cv2
import numpy as np
import torch
from detectron2.data.transforms import ResizeShortestEdge
from PIL import Image
from RandAugment import RandAugment
from torch.utils.data import Dataset
from torchvision import transforms

from .build import DATASETS_REGISTRY

__all__ = ['COCOCaptions']

class COCOCaptions(Dataset):
    def __init__(self,
                 root,
                 annotations_file,
                 images_dir,
                 transform=None,
                 limit_edge=None,
                 worker_fn=None):
        super(COCOCaptions, self).__init__()
        self.root             = root
        self.annotations_file = os.path.join(self.root, annotations_file)
        self.images_dir       = os.path.join(self.root, images_dir)
        self.worker_fn        = worker_fn
        self.worker_fn        = worker_fn
        self.transform        = transform
        self.limit_edge       = ResizeShortestEdge(*limit_edge) if limit_edge is not None else None

        with open(self.annotations_file, 'r') as fp:
            raw_data = json.load(fp)
        self.name     = 'COCO'
        self.info     = raw_data['info']['description']
        #self.id2img   = {item['id']: item['file_name'] for item in raw_data['images']}
        self.dataset  = raw_data['annotations']

    def __getitem__(self, index):
        ret_dict            = {}
        row                 = self.dataset[index]
        ret_dict['caption'] = row['caption']
        ret_dict['id']      = row['id']
        img_name            = '{}.jpg'.format(row['image_id'])#self.id2img[row['image_id']]
        img_path            = os.path.join(self.images_dir, img_name)

        #h,w,c in BGR format (PIL opens it in RGB format) -> convert to RGB
        img = Image.open(img_path).convert('RGB')
        #img.save('original.jpg'); self.transform.transforms[0](img).save('augment.jpg')
        if self.transform is not None:
            #resize shortest edge and limit the longest one while keeping the aspect ratio.
            img = self.transform(img)
            if self.limit_edge is not None:
                #randaugment works with PIL. ResizeShortest works with numpy.ndarray
                img = np.array(img.permute(1, 2, 0)) #correct format for ResizeShortest
                img = self.limit_edge.get_transform(img).apply_image(img)
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) #from float64 to float32 to allow mixed precision
        ret_dict['img'] = img
        return ret_dict if self.worker_fn is None else self.worker_fn(ret_dict)

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return self.info
    

@DATASETS_REGISTRY.register()
def build_coco_dataset(cfg, **kwargs):
    tr = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(cfg.MEAN, cfg.STD)])
    if cfg.has_attr('RANDAUGMENT'):
        randaug = RandAugment(*cfg.RANDAUGMENT)
        #remove color inversion, posterize, solarize, solarize add, cutout
        for i in reversed([2, 4, 5, 6, 13]):
            randaug.augment_list.pop(i)
        tr.transforms.insert(0, randaug)
    #Resize before RandAugment
    if cfg.has_attr('RESIZE'):
        tr.transforms.insert(0, transforms.Resize(*cfg.RESIZE))
    trainset = COCOCaptions(root=cfg.ROOT,
                            annotations_file=cfg.TRAIN[0],
                            images_dir=cfg.IMAGES_DIR,
                            transform=tr,
                            limit_edge=cfg.LIMIT_EDGE if cfg.has_attr('LIMIT_EDGE') else None,
                            worker_fn=kwargs['worker_fn'],
                        )
    devset = COCOCaptions(root=cfg.ROOT,
                          annotations_file=cfg.DEV[0],
                          images_dir=cfg.IMAGES_DIR,
                          transform=tr,
                          limit_edge=cfg.LIMIT_EDGE if cfg.has_attr('LIMIT_EDGE') else None,
                          worker_fn=kwargs['worker_fn'] if 'worker_fn' in kwargs else None
                        )
    return trainset, devset
