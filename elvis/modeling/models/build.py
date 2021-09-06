import pdb

from fvcore.common.registry import Registry  # noqa

__all__ = ['NET_REGISTRY',
           'build_net']

NET_REGISTRY = Registry('MULTIMODAL')

def build_net(cfg, get_interface=False, **kwargs):
    return NET_REGISTRY.get(cfg.NET.NAME)(cfg, get_interface=get_interface, **kwargs)