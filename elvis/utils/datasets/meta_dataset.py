import pdb
from typing import List

from torch.utils.data import Dataset

from .build import DATASETS_REGISTRY, build_dataset


class MetaDataset(Dataset):
    def __init__(self, d_list: List):
        super(MetaDataset, self).__init__()
        self.datasets = {d.name: d for d in d_list}
        self.meta_idx = []
        for dataset in d_list:
            self.meta_idx.extend({'name': dataset.name, 'index': idx} for idx in range(len(dataset)))

    def __getitem__(self, index):
        meta_row = self.meta_idx[index]
        row      = self.datasets[meta_row['name']][meta_row['index']]
        return meta_row['name'], row

    def __len__(self):
        return len(self.meta_idx)

    def __str__(self):
        print([d for d in self.datasets.keys()])


@DATASETS_REGISTRY.register()
def build_meta_dataset(cfg, **kwargs):
    train_d_list = []
    dev_d_list   = []
    for dataset in cfg.DATASETS:
        train_d, dev_d = build_dataset(dataset, worker_fn=None)
        train_d_list.append(train_d)
        dev_d_list.append(dev_d)
    trainset = MetaDataset(train_d_list)
    devset   = MetaDataset(dev_d_list)
    return trainset, devset

