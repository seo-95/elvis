import pdb

from fvcore.common.registry import Registry  # noqa

__all__ = ['VIT_BERT_INTERFACE_REGISTRY',
           'build_net']

VIT_BERT_INTERFACE_REGISTRY = Registry('VIT_BERT')

def build_data_interface(interface_name, **kwargs):
    call_name = 'build_{}_interface'.format(interface_name)
    return VIT_BERT_INTERFACE_REGISTRY.get(call_name)(**kwargs)