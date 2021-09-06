import copy
import os
import pdb
import random
from typing import Dict, List, Text, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from elvis.modeling.models import build_net
from elvis.modeling.models.layers import FC, MLP
from elvis.utils.vlp_objectives import optimal_transport_dist

from .base import MetaArch
from .build import ARCH_REGISTRY

Tensor = TypeVar('torch.tensor')


__all__ = ['AlignmentVLP',
           'build_align_vlp']


class AlignmentVLP(MetaArch):
    """Meta architecture for Visual Language Pretraining (VLP) based on image-caption alignment
    """
    def __init__(self, model, max_visual, max_tokens, tasks_dict) -> None:
        super().__init__()
        self.model       = model
        self.max_visual  = max_visual
        self.max_tokens  = max_tokens+2 #take into account [CLS] and [SEP]
        self.tasks_dict  = tasks_dict

        self.lm_mlp = MLP(in_features=self.model.embed_dim,
                          hidden_dim=self.model.embed_dim,
                          out_features=len(self.model.tokenizer)-1,
                          dropout_p=.1)
        self.itm_fc = FC(in_features=self.model.embed_dim, out_features=2)
    
    def forward(self, vis_in, txt_in, vis_mask, txt_mask, **kwargs) -> Dict:
        cntx_emb   = self.model(vis_in=vis_in, vis_mask=vis_mask, txt_in=txt_in, txt_mask=txt_mask)
        txt_emb    = cntx_emb[:, :self.max_tokens]

        itm_logits = self.itm_fc(txt_emb[:, 0, :]) #pass everything but use only [CLS]: better parallelization of loss computation
        lm_logits  = self.lm_mlp(txt_emb[:, 1:, :])

        #? exclude special tokens from ot computation
        vis_mask = torch.cat(
                            (torch.ones((vis_mask.shape[0], 1), device=vis_mask.device), vis_mask),
                            dim=-1) #add attention for [IMG]
        ot_dist = optimal_transport_dist(txt_emb=cntx_emb[:, :self.max_tokens, :].float(),
                                         img_emb=cntx_emb[:, self.max_tokens:, :].float(),
                                         txt_pad=~txt_mask.bool(),
                                         img_pad=~vis_mask.bool()
                                        )

        return {'lm_logits': lm_logits, 'itm_logits': itm_logits, 'ot_dist': ot_dist}

    def compute_loss(self, lm_logits, itm_logits, lm_targets, itm_targets, **kwargs) -> Dict:
        B = lm_logits.shape[0]
        n_mlm      = sum([t == 'MLM' for t in kwargs['tasks']])
        n_itm      = len(kwargs['tasks']) - n_mlm
        loss_dict  = {}

        #compute lm loss (compute it also if n_mlm > 0 otherwise the DDP will raise an exception)
        lm_loss  = F.cross_entropy(lm_logits.transpose(1, 2), lm_targets[:, 1:], reduction='sum')
        if n_mlm > 0:
            lm_loss /= n_mlm
        loss_dict['lm_loss'] = lm_loss

        #compute itm loss (compute it also if n_itm > 0 otherwise the DDP will raise an exception)
        itm_loss = F.cross_entropy(itm_logits, itm_targets[:, 0], reduction='sum')
        ot_pos   = kwargs['ot_dist'].masked_select(itm_targets[:, 0] == 1)
        ot_neg   = kwargs['ot_dist'].masked_select(itm_targets[:, 0] == 0)
        #we want to maximize the OT distance for negative pairs and minimize OT distance for positive ones
        ot_loss  = ot_pos.sum() - ot_neg.sum()
        itm_loss = (itm_loss + 0.1 * ot_loss)
        if n_itm > 0:
            itm_loss /= n_itm
        loss_dict['itm_loss'] = itm_loss

        loss_dict['loss'] = sum(loss_dict.values())
        return loss_dict

    def save_on_disk(self, path):
        state_dict = copy.deepcopy(self).cpu().state_dict()
        ckp_file   = os.path.join(path, 'state_dict.pt')
        torch.save(state_dict, ckp_file)



@ARCH_REGISTRY.register()
def build_align_vlp(cfg):
    model, data_interface = build_net(cfg.MODEL, get_interface='vlp')
    vlp                   = AlignmentVLP(model,
                                         max_visual=cfg.MODEL.MAX_N_VISUAL,
                                         max_tokens=cfg.MODEL.MAX_N_TOKENS,
                                         tasks_dict=cfg.MODEL.TASKS.get_as_dict())
    return vlp, data_interface

