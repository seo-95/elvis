import pdb

from fvcore.common.registry import Registry  # noqa

__all__ = ['VILT_INTERFACE_REGISTRY',
           'build_net']

VILT_INTERFACE_REGISTRY = Registry('VILT')

def build_data_interface(interface_name, **kwargs):
    call_name = 'build_{}_interface'.format(interface_name)
    return VILT_INTERFACE_REGISTRY.get(call_name)(**kwargs)