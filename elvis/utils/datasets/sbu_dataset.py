import json
import os
import pdb
import random

import cv2
import torch
from detectron2.data.transforms import ResizeShortestEdge
from torch.utils.data import Dataset

from .build import DATASETS_REGISTRY, VoidDataset

__all__ = ['SBUDataset']

class SBUDataset(Dataset):
    def __init__(self,
                 root,
                 annotations_file,
                 images_dir,
                 worker_fn=None):
        super(SBUDataset, self).__init__()
        self.root             = root
        self.annotations_file = os.path.join(self.root, annotations_file)
        self.images_dir       = os.path.join(self.root, images_dir)
        self.worker_fn        = worker_fn

        with open(self.annotations_file, 'r') as fp:
            raw_data = json.load(fp)
        self.name    = 'SBU'
        self.info    = raw_data['name']
        self.dataset = raw_data['annotations']


    def __getitem__(self, index):
        ret_dict            = {}
        row                 = self.dataset[index]
        ret_dict['caption'] = row['caption']
        ret_dict['id']      = row['id']
        img_name            = row['image']
        img_path            = os.path.join(self.images_dir, img_name)
        #h,w,c in BGR format (PIL opens it in RGB format) -> convert to RGB
        img                 = cv2.imread(img_path)
        img                 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resize shortest edge and limit the longest one while keeping the aspect ratio.
        img                 = ResizeShortestEdge(384, 640).get_transform(img).apply_image(img)
        ret_dict['img']     = torch.as_tensor(img.astype('float32').transpose(2, 0, 1))
        
        return ret_dict if self.worker_fn is None else self.worker_fn(ret_dict)

    def __str__(self):
        return self.info

    def __len__(self):
        return len(self.dataset)


@DATASETS_REGISTRY.register()
def build_sbu_dataset(cfg, **kwargs):
    trainset = SBUDataset(root=cfg.ROOT,
                          annotations_file=cfg.TRAIN[0],
                          images_dir=cfg.IMAGES_DIR,
                          worker_fn=kwargs['worker_fn'],
                        )
    if 'DEV' in cfg.get_as_dict():
        devset = SBUDataset(root=cfg.ROOT,
                            annotations_file=cfg.DEV[0],
                            images_dir=cfg.IMAGES_DIR,
                            worker_fn=kwargs['worker_fn']
                            )
    else:
        devset      = VoidDataset()
        devset.name = 'Void VG'
    return trainset, devset