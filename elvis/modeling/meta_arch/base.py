from abc import abstractmethod
from typing import Any, List, Tuple, TypeVar, Dict

import torch.nn as nn

Tensor = TypeVar('torch.tensor')


__all__ = ['MetaArch']


class MetaArch(nn.Module):
    def __init__(self) -> None:
        super(MetaArch, self).__init__()

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, *inputs: Any, **kwargs) -> Dict:
        pass

    """
    @abstractmethod
    def predict(self, inputs):
            Inference with single elements
        raise NotImplementedError
    """

    @abstractmethod
    def save_on_disk(self, path):
        raise NotImplementedError

    @abstractmethod
    def from_checkpoint(self, path):
        raise NotImplementedError