import copy
import json
import os
import pdb
import re
from typing import Dict, List, TypeVar

import torch
from elvis.modeling.models import build_net
from elvis.modeling.models.layers import MLP
from torch.nn import functional as F

from .base import MetaArch
from .build import ARCH_REGISTRY

Tensor  = TypeVar('torch.tensor')


__all__ = ['MetaVQA',
           'build_meta_vqa']

class MetaVQA(MetaArch):

    def __init__(self,
                model,
                max_visual,
                max_tokens,
                ans2id):
        super(MetaArch, self).__init__()
        self.model       = model
        self.max_visual  = max_visual
        self.max_tokens  = max_tokens
        self.ans2id      = ans2id
        self.id2ans      = {v: k for k, v in ans2id.items()}
        self.out_layer   = MLP(in_features=self.model.embed_dim,
                               hidden_dim=self.model.embed_dim,
                               out_features=len(self.ans2id),
                               dropout_p=.1)

    def forward(self, vis_in, txt_in, vis_mask, txt_mask, **kwargs):
        out = self.model(vis_in=vis_in, vis_mask=vis_mask, txt_in=txt_in, txt_mask=txt_mask)
        t_pool = out[:, 0]
        #v_pool = out[:, self.max_tokens]
        logits = self.out_layer(t_pool)
        return {'vqa_logits': logits}

    def compute_loss(self, vqa_logits, gt_answers, **kwargs) -> Dict:
        vqa_loss = F.binary_cross_entropy_with_logits(vqa_logits, gt_answers, reduction='none')
        vqa_loss = vqa_loss.sum(dim=-1).mean()
        return {'loss': vqa_loss}

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

    def from_pretrained(self, state_dict):
        layers_names = list(state_dict.keys())
        for l_name in layers_names:
            if l_name.startswith('lm_mlp') or l_name.startswith('itm_fc'):
                del state_dict[l_name]
            else:
                #remove the model.layer_name
                state_dict[l_name[6:]] = state_dict.pop(l_name)
        self.model.load_state_dict(state_dict)

    def predict(self, vis_in, txt_in, vis_mask, txt_mask, **kwargs):
        out         = self.forward(vis_in, txt_in, vis_mask, txt_mask, **kwargs)
        probs       = torch.sigmoid(out['vqa_logits']).squeeze(0)
        answer_id   = torch.argmax(probs).item()
        answer_conf = probs[answer_id].item() 
        answer      = self.id2ans[answer_id]
        return answer, answer_conf

    """
    def from_checkpoint(self, path):
        self.lang_net.load_config(path)
        state_path = os.path.join(path, 'state_dict.pt')
        state_dict = torch.load(state_path)
        self.load_state_dict(state_dict)

        voc_path = os.path.join(path, 'label2ans.json')
        with open(voc_path) as fp:
            self.id2ans = json.load(fp)
    """




@ARCH_REGISTRY.register()
def build_meta_vqa(cfg, **kwargs):
    with open(cfg.MODEL.ANS_VOCAB) as fp:
        ans2id = json.load(fp)

    model, data_interface = build_net(cfg.MODEL, get_interface='vqa', **{'ans2id': ans2id})
    vqa   =  MetaVQA(model,
                     max_visual=cfg.MODEL.MAX_N_VISUAL,
                     max_tokens=cfg.MODEL.MAX_N_TOKENS,
                     ans2id=ans2id)
    return vqa, data_interface
