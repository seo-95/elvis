import pdb
from collections import Counter
from typing import Dict, List, Text, TypeVar

import torch
from elvis.modeling.models.base import ModelDataInterface

from .dispatcher import VILT_INTERFACE_REGISTRY


class ViLTVQADataInterface(ModelDataInterface):
    def __init__(self, max_patches, max_tokens, tokenizer, patch_size, prepare_patches_fn, ans2id):
        self.prepare_patches_fn = prepare_patches_fn
        self.max_patches        = max_patches
        self.max_tokens         = max_tokens
        self.tokenizer          = tokenizer
        self.patch_size         = patch_size
        self.ans2id             = ans2id

    def worker_fn(self, x):
        # dict_keys(['quest_id', 'img_id', 'quest', 'true_ans', 'ans_list', 'img'])
        patches      = self.prepare_patches_fn(x['img'], self.patch_size)
        patches      = patches[:self.max_patches]
        patches_mask = torch.ones(patches.shape[0])
        to_pad       = self.max_patches - patches.shape[0]
        if to_pad > 0:
            padding      = torch.zeros(to_pad, patches.shape[-1])
            patches      = torch.cat((patches, padding), dim=0)
            padding      = torch.zeros(to_pad)
            patches_mask = torch.cat((patches_mask, padding), dim=0)

        tok_res  = self.tokenizer(x['quest'], padding='max_length', truncation=True, max_length=self.max_tokens+2, return_tensors='pt')
        ids      = tok_res['input_ids'].squeeze(0)
        ids_mask = tok_res['attention_mask'].squeeze(0)

        #preprocess answer
        ans_targets = torch.zeros(len(self.ans2id))
        # x['ans_list'] is a tuple (ans, confidence)
        support_dict    = Counter([ans[0] for ans in x['ans_list']])
        ans_soft_scores = [(ans, min(supp/3, 1.0)) for ans, supp in support_dict.items()]
        for ans, score in ans_soft_scores:
            if ans in self.ans2id:
                ans_targets[self.ans2id[ans]] = score

        assert patches.shape[0] == self.max_patches and ids.shape[0] == self.max_tokens+2
        ret_dict = {}
        ret_dict['patches']      = patches
        ret_dict['patches_mask'] = patches_mask
        ret_dict['ids']          = ids
        ret_dict['ids_mask']     = ids_mask
        ret_dict['gt_answers']   = ans_targets
        ret_dict['sample_id']    = x['quest_id']

        return ret_dict

    def worker_fn_eval(self, x):
        #dict_keys(['image_id', 'img', 'question', 'question_id'])
        patches      = self.prepare_patches_fn(x['img'], self.patch_size)
        patches      = patches[:self.max_patches]
        patches_mask = torch.ones(patches.shape[0])
        tok_res      = self.tokenizer(x['question'], truncation=True, max_length=self.max_tokens+2, return_tensors='pt')
        ids          = tok_res['input_ids']
        ids_mask     = tok_res['attention_mask']

        ret_dict = {}
        ret_dict['question_id'] = x['question_id']
        ret_dict['image_id']    = x['image_id']
        ret_dict['vis_in']      = patches.unsqueeze(0)
        ret_dict['vis_mask']    = patches_mask.unsqueeze(0)
        ret_dict['txt_in']      = ids
        ret_dict['txt_mask']    = ids_mask

        return ret_dict
        

    def collate_fn(self, batch):
        ret_dict = {}
        sample_ids   = [item['sample_id'] for item in batch]
        patches      = [item['patches'] for item in batch]
        patches_mask = [item['patches_mask'] for item in batch]
        ids          = [item['ids'] for item in batch]
        ids_mask     = [item['ids_mask'] for item in batch]
        answers      = [item['gt_answers'] for item in batch]

        ret_dict = {'samples_id': sample_ids,
                    'vis_in'    : torch.stack(patches),
                    'vis_mask'  : torch.stack(patches_mask),
                    'txt_in'    : torch.stack(ids),
                    'txt_mask'  : torch.stack(ids_mask),
                    'gt_answers': torch.stack(answers)
                    }
        return ret_dict





@VILT_INTERFACE_REGISTRY.register()
def build_vqa_interface(cfg, tokenizer, patch_size, prepare_patches_fn, ans2id):
    return ViLTVQADataInterface(max_patches=cfg.MAX_N_VISUAL,
                                max_tokens=cfg.MAX_N_TOKENS,
                                tokenizer=tokenizer,
                                patch_size=patch_size,
                                prepare_patches_fn=prepare_patches_fn,
                                ans2id=ans2id)
