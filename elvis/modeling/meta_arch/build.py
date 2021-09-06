import pdb

from fvcore.common.registry import Registry  # noqa

__all__ = ['ARCH_REGISTRY',
           'build_model']

ARCH_REGISTRY = Registry('META_ARCH')


def build_model(cfg, **kwargs):
    return ARCH_REGISTRY.get(cfg.MODEL.NAME)(cfg, **kwargs)