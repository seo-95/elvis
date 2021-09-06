import json
import os
import pdb

import numpy as np
import torch
from detectron2.data.transforms import ResizeShortestEdge
from PIL import Image
from RandAugment import RandAugment
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor

from .build import DATASETS_REGISTRY, VoidDataset

__all__ = ['VGCaptions']

class VGCaptions(Dataset):
    def __init__(self,
                 root,
                 annotations_file,
                 images_dir,
                 transform=None,
                 limit_edge=None,
                 worker_fn=None):
        super(VGCaptions, self).__init__()
        self.root             = root
        self.annotations_file = os.path.join(self.root, annotations_file)
        self.images_dir       = os.path.join(self.root, images_dir)
        self.worker_fn        = worker_fn
        self.transform        = transform
        self.limit_edge       = ResizeShortestEdge(*limit_edge) if limit_edge is not None else None

        with open(self.annotations_file, 'r') as fp:
            raw_data = json.load(fp)
        self.dataset = []
        for anns in raw_data:
            for reg_ann in anns['regions']:
                self.dataset.append(reg_ann)
        self.name     = 'VG'
        self.info     = 'VG captioning dataset'

    def __getitem__(self, index):
        ret_dict              = {}
        row                   = self.dataset[index]
        ret_dict['caption']   = row['phrase']
        ret_dict['region_id'] = row['region_id']
        img_id                = row['image_id']
        w, h                  = row['width'], row['height']
        x, y                  = row['x'], row['y']
        #in VG some annotations is wrongly reported as been negative
        x *= -1 if x < 0 else 1
        y *= -1 if y < 0 else 1
        img_path = os.path.join(self.images_dir, '{}.jpg'.format(img_id))
        #h,w,c in BGR format (PIL opens it in RGB format)
        img = Image.open(img_path).convert('RGB')

        #draw rectangle:
        #img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2); cv2.imwrite("my.png",img)
        #here return only the region described by the captions
        #img = img[y:y+h, x:x+w, :]
        #img.save('original.jpg'); self.transform.transforms[0](img).save('augment.jpg')

        if self.transform is not None:
            #resize shortest edge and limit the longest one while keeping the aspect ratio.
            img  = self.transform(img)
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
def build_vg_captions_dataset(cfg, **kwargs):
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.MEAN, cfg.STD)])
    if cfg.has_attr('RANDAUGMENT'):
        randaug = RandAugment(*cfg.RANDAUGMENT)
        #remove color inversion, posterize, solarize, solarize add, cutout
        for i in reversed([2, 4, 5, 6, 13]):
            randaug.augment_list.pop(i)
        tr.transforms.insert(0, randaug)
    #Resize before RandAugment
    if cfg.has_attr('RESIZE'):
        tr.transforms.insert(0, transforms.Resize(*cfg.RESIZE))
    trainset = VGCaptions(root=cfg.ROOT,
                          annotations_file=cfg.TRAIN[0],
                          images_dir=cfg.IMAGES_DIR,
                          limit_edge=cfg.LIMIT_EDGE if cfg.has_attr('LIMIT_EDGE') else None,
                          transform=tr,
                          worker_fn=kwargs['worker_fn'] if 'worker_fn' in kwargs else None
                        )
    if 'DEV' in cfg.get_as_dict():
        devset   = VGCaptions(root=cfg.ROOT,
                            annotations_file=cfg.DEV[0],
                            images_dir=cfg.IMAGES_DIR,
                            resize_shortest=cfg.LIMIT_EDGE if cfg.has_attr('LIMIT_EDGE') else None,
                            transform=tr,
                            worker_fn=kwargs['worker_fn'] if 'worker_fn' in kwargs else None
                            )
    else:
        devset      = VoidDataset()
        devset.name = 'Void SBU'
    return trainset, devset
