import pdb

import yaml

class ConfigNode(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [ConfigNode(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, ConfigNode(b) if isinstance(b, dict) else b)

    def save_to_disk(self, save_path):
        with open(save_path, 'w') as fp:
            yaml.dump(self.get_as_dict(self), fp, default_flow_style=False)

    def get_as_dict(self, node=None):
        if node is None:
            node = self
        res_dict = {}
        for a, b in node.__dict__.items():
            res_dict[a] = self.get_as_dict(b) if isinstance(b, ConfigNode) else b
        return res_dict

    def has_attr(self, attr: str) -> bool:
        """Return true if the attribute is inside the first level of this node

        Args:
            attr (str): the name of the attribute to search

        Returns:
            bool: True if the attribute is inside the first level of this node (False otherwise)
        """
        return attr in self.__dict__

    def __str__(self):
        return str(self.get_as_dict(self))
