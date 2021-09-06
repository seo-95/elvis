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
from VQAtools import VQA, VQAEval

__all__ = ['VQADataset']


class VQADataset(Dataset):

    def __init__(self, 
                root, 
                questions_file, 
                annotations_file,
                images_dir,
                transform=None,
                limit_edge=None,
                worker_fn=None):
        """
        Wrapper for VQA 2.0 dataset
        Args:
            data_folder (str): path to VQA folder (annotations if available + questions)
            visual_features_file (str): path to HDF5 or torch file containing visual features
            split (str, optional): dataset split ('train', 'val', 'test'). Defaults to 'train'.
        """
        super(VQADataset, self).__init__()
        #q2q -> questions
        #q2a -> question annotations

        self.root            = root
        self.images_dir      = os.path.join(root, images_dir)
        self.question_file   = os.path.join(root, questions_file)
        self.annotation_file = os.path.join(root, annotations_file)
        self.worker_fn       = worker_fn
        self.transform       = transform
        self.limit_edge      = ResizeShortestEdge(*limit_edge) if limit_edge is not None else None

        self.dataset    = VQA(annotation_file=self.annotation_file,  
                              question_file=self.question_file)
        
        self.indices = [quest_id for quest_id in self.dataset.q2q.keys()]

    def __getitem__(self, index):
        ret_dict = {}
        quest_id = self.indices[index]

        ret_dict['quest_id'] = quest_id
        ret_dict['img_id']   = str(self.dataset.q2q[quest_id]['image_id'])
        ret_dict['quest']    = self.dataset.q2q[quest_id]['question']
        ret_dict['true_ans'] = self.dataset.q2a[quest_id]['multiple_choice_answer']
        ret_dict['ans_list'] = [(ii['answer'], ii['answer_confidence']) for ii in self.dataset.q2a[quest_id]['answers']]
        img_name = '{}.jpg'.format(ret_dict['img_id'])
        img_path = os.path.join(self.images_dir, img_name)

        #h,w,c in BGR format (PIL opens it in RGB format) -> convert to RGB
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            if self.limit_edge is not None:
                #randaugment works with PIL. ResizeShortest works with numpy.ndarray
                img = np.array(img.permute(1, 2, 0))
                img = self.limit_edge.get_transform(img).apply_image(img)
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) #from float64 to float32 to allow mixed precision
        ret_dict['img'] = img
        return ret_dict if self.worker_fn is None else self.worker_fn(ret_dict)

    def compute_accuracy(self, res_list, logging_dir):
        #logging_dir is used to save the tmp file in a safe place to avoid conflicts with other trainings
        tmp_file = os.path.join(logging_dir, 'tmp_results.json')
        with open(tmp_file, 'w') as fp:
            json.dump(res_list, fp)
        vqaRes  = self.dataset.loadRes(resFile=tmp_file, quesFile=self.question_file)
        vqaEval = VQAEval(self.dataset, vqaRes, n=2)
        os.remove(tmp_file)
        return vqaEval.evaluate()

    def __len__(self):
        return len(self.indices)

    def __str__(self) -> str:
        return self.dataset.info()


from .build import DATASETS_REGISTRY


@DATASETS_REGISTRY.register()
def build_vqa_dataset(cfg, **kwargs):
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
        tr.transforms.insert(0, transforms.Resize(size=cfg.RESIZE))

    trainset =  VQADataset(root=cfg.ROOT,
                           questions_file=cfg.TRAIN[0],
                           annotations_file=cfg.TRAIN[1],
                           images_dir=cfg.IMAGES_DIR,
                           transform=tr,
                           limit_edge=cfg.LIMIT_EDGE if cfg.has_attr('LIMIT_EDGE') else None, 
                           worker_fn=kwargs['worker_fn']
                        )
    devset  = VQADataset(root=cfg.ROOT,
                         questions_file=cfg.DEV[0],
                         annotations_file=cfg.DEV[1],
                         images_dir=cfg.IMAGES_DIR,
                         transform=tr,
                         limit_edge=cfg.LIMIT_EDGE if cfg.has_attr('LIMIT_EDGE') else None, 
                         worker_fn=kwargs['worker_fn']
                        )
    return (trainset, devset)
    
