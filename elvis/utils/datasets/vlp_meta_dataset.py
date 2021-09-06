import copy
import pdb
import random

import torch

from .build import DATASETS_REGISTRY, build_dataset
from .meta_dataset import MetaDataset


class VLPDataset(MetaDataset):
    def __init__(self, d_list, worker_fn=None):
        super(VLPDataset, self).__init__(d_list)
        self.worker_fn = worker_fn

    def __getitem__(self, index):
        ret_dict = self._get_sample(index)
        #here extract a random sample 50% of the times
        neg_sample = None
        if random.uniform(0, 1) > .5:
            neg_idx = ret_dict['id']
            while neg_idx == ret_dict['id']:
                neg_idx = random.randint(0, len(self)-1)
            neg_sample = self._get_sample(neg_idx)
        
        return self.worker_fn(ret_dict, neg_sample)

    def _get_sample(self, index):
        ret_dict = {}
        d_name, row = super().__getitem__(index)
        ret_dict['id']      = row['region_id'] if d_name == 'VG' else row['id']
        ret_dict['text']    = row['caption']
        ret_dict['img']     = row['img']
        ret_dict['dataset'] = d_name
        #print('name: {}, id: {}'.format(d_name, ret_dict['id']))
        return ret_dict


@DATASETS_REGISTRY.register()
def build_vlp_meta_dataset(cfg, **kwargs):
    train_d_list = []
    dev_d_list   = []
    for dataset in cfg.DATASETS:
        dataset_cfg              = copy.deepcopy(dataset)
        if cfg.has_attr('RESIZE_EDGE'):
            dataset_cfg.RESIZE_EDGE  = cfg.RESIZE_EDGE #inherit param
        if cfg.has_attr('RANDAUGMENT'):
            dataset_cfg.RANDAUGMENT  = cfg.RANDAUGMENT #inherit param
        dataset_cfg.MEAN         = cfg.MEAN
        dataset_cfg.STD          = cfg.STD
        train_d, dev_d           = build_dataset(dataset_cfg, worker_fn=None)
        train_d_list.append(train_d)
        dev_d_list.append(dev_d)
    trainset = VLPDataset(train_d_list, worker_fn=kwargs['worker_fn'])
    devset   = VLPDataset(dev_d_list, worker_fn=kwargs['worker_fn'])
    return trainset, devset
