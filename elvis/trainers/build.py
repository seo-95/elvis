import pdb

from fvcore.common.registry import Registry  # noqa

__all__ = ['TRAINER_REGISTRY',
           'build_trainer']

TRAINER_REGISTRY = Registry('TRAINER_ARCH')


def build_trainer(cfg, **kwargs):
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg, **kwargs)