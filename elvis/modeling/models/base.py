from abc import abstractmethod
from typing import Any, Dict, List, Tuple, TypeVar

Tensor = TypeVar('torch.tensor')


__all__ = ['ModelDataInterface']


class ModelDataInterface(object):
    """This class represents the common interface to connect the Dataset and the MetaArch.
    The 2 methods present in this class are used to allow the communication between the two components.
    """
    def __init__(self):
        pass

    def worker_fn(self):
        """This function is executed inside each worker

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def collate_fn(self, batch):
        """This function is executed on the main process and is responsible to ensamble the batch
        """
        raise NotImplementedError
