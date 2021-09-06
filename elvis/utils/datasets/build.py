import sys
import pdb

from torch.utils.data.dataset import Dataset

sys.path.append('.')

from fvcore.common.registry import Registry  # noqa


DATASETS_REGISTRY = Registry('DATASETS')


class VoidDataset(Dataset):
    def __init__(self):
        super(VoidDataset, self).__init__()
    def __getitem__(self, index):
        pass
    def __len__(self):
        return 0


def build_dataset(cfg, **kwargs):
#    return DATASETS_REGISTRY.get(cfg.TRAINER.DATASET.NAME)(cfg.TRAINER.DATASET, **kwargs)
    return DATASETS_REGISTRY.get(cfg.NAME)(cfg, **kwargs)