import copy
import json
import os
import pdb
import re
from typing import Dict, List, TypeVar

import torch
from elvis.modeling.models import build_net
from elvis.modeling.models.layers import FC
from torch.nn import functional as F

from .base import MetaArch
from .build import ARCH_REGISTRY

Tensor  = TypeVar('torch.tensor')


__all__ = ['MetaRetrieval',
           'build_meta_retrieval']

class MetaRetrieval(MetaArch):

    def __init__(self,
                model,
                max_patches,
                max_tokens):
        super(MetaArch, self).__init__()
        self.model       = model
        self.max_patches = max_patches
        self.max_tokens  = max_tokens

        self.itm_fc = FC(in_features=self.model.embed_dim, out_features=2)

    def forward(self, vis_in, txt_in, vis_mask, txt_mask, **kwargs):
        out = self.model(vis_in=vis_in, vis_mask=vis_mask, txt_in=txt_in, txt_mask=txt_mask)
        t_pool = out[:, 0]
        #v_pool = out[:, self.max_tokens]
        logits = self.itm_fc(t_pool)
        return {'retrieval_logits': logits}

    def compute_loss(self, vqa_logits, gt_answers, **kwargs) -> Dict:
        pass #todo
        """
        vqa_loss = F.binary_cross_entropy_with_logits(vqa_logits, gt_answers, reduction='none')
        vqa_loss = vqa_loss.sum(dim=-1).mean()
        return {'loss': vqa_loss}
        """

    def save_on_disk(self, path):
        #save vocab only once
        vocab_ckp = os.path.join(path, 'VQA.vocab')
        if not os.path.exists(vocab_ckp):
            with open(vocab_ckp, 'w') as fp:
                json.dump(self.ans2id, fp)

        #use deepcopy to avoid problems with DistributedDataParallel
        state_dict = copy.deepcopy(self).cpu().state_dict()
        ckp_file = os.path.join(path, 'state_dict.pt')
        torch.save(state_dict, ckp_file)

    def predict(self, vis_in, txt_in, vis_mask, txt_mask, **kwargs):
        out         = self.forward(vis_in, txt_in, vis_mask, txt_mask, **kwargs)
        probs       = F.softmax(out['retrieval_logits'], dim=-1).squeeze(0)
        #return score of item being the true one
        if probs.dim() > 1:
            scores = [p[1].item() for p in probs]
            return scores
        score       = probs[1].item()
        return score




@ARCH_REGISTRY.register()
def build_meta_retrieval(cfg, **kwargs):
    model, data_interface = build_net(cfg.MODEL, get_interface='retrieval')
    vqa   =  MetaRetrieval(model,
                           max_patches=cfg.MODEL.MAX_VIS_PATCHES,
                           max_tokens=cfg.MODEL.MAX_N_TOKENS)
    return vqa, data_interface
