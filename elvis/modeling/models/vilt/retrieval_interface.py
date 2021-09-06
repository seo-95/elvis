import pdb
from typing import Dict, List, Text, TypeVar

import torch
from elvis.modeling.models.base import ModelDataInterface

from .dispatcher import VILT_INTERFACE_REGISTRY


class ViLTRetrievalInterface(ModelDataInterface):
    def __init__(self, max_patches, max_tokens, tokenizer, patch_size, prepare_patches_fn):
        self.prepare_patches_fn = prepare_patches_fn
        self.max_patches        = max_patches
        self.max_tokens         = max_tokens
        self.tokenizer          = tokenizer
        self.patch_size         = patch_size

    def worker_fn(self, x):
        pass #todo
        """
        # dict_keys(['quest_id', 'img_id', 'quest', 'ans', 'img'])
        patches      = self.prepare_patches_fn(x['img'], self.patch_size)
        patches      = patches[:self.max_patches]
        patches_mask = torch.ones(patches.shape[0])
        to_pad       = self.max_patches - patches.shape[0]
        if to_pad > 0:
            padding      = torch.zeros(to_pad, patches.shape[-1])
            patches      = torch.cat((patches, padding), dim=0)
            padding      = torch.zeros(to_pad)
            patches_mask = torch.cat((patches_mask, padding), dim=0)
        tok_res        = self.tokenizer(x['quest'], padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors='pt')
        ids            = tok_res['input_ids'].squeeze(0)
        ids_mask       = tok_res['attention_mask'].squeeze(0)
        #preprocess answer
        ans_targets    = torch.zeros(len(self.ans2id))
        ans_ids = set()
        for ans in x['ans']:
            text = ans[0]
            conf = ans[1]
            if conf != 'no':
                ans_ids.add(self.ans2id[text] if text in self.ans2id else self.ans2id['<UNK>'])
        ans_targets[list(ans_ids)] = 1

        assert patches.shape[0] == self.max_patches and ids.shape[0] == self.max_tokens
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
        patches_mask = torch.ones(patches.shape[0])
        tok_res      = self.tokenizer(x['question'], truncation=True, max_length=self.max_tokens, return_tensors='pt')
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
        """
        

    def collate_fn(self, batch):
        pass #todo
        """
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
        """

    def worker_fn_eval(self, x):
        #dict_keys(['id', 'caption', 'imgs', 'image_ids])
        patches       = [self.prepare_patches_fn(img, self.patch_size)[:self.max_patches] for img in x['imgs']]
        patches = patches[:500] #todo remove
        patches_masks = [torch.ones(curr_patches.shape[0]) for curr_patches in patches]
        for idx in range(len(patches)):
            to_pad = self.max_patches - patches[idx].shape[0]
            if to_pad > 0:
                padding            = torch.zeros(to_pad, patches[idx].shape[-1])
                patches[idx]       = torch.cat((patches[idx], padding), dim=0)
                padding            = torch.zeros(to_pad) 
                patches_masks[idx] = torch.cat((patches_masks[idx], padding), dim=0)
        tok_res  = self.tokenizer(x['caption'], truncation=True, max_length=self.max_tokens+2, return_tensors='pt')
        ids      = tok_res['input_ids']
        ids_mask = tok_res['attention_mask']

        ret_dict = {}
        ret_dict['id']       = x['id']
        ret_dict['image_id'] = x['image_ids']
        ret_dict['vis_in']   = torch.stack(patches).cuda() #always on gpu during inference
        ret_dict['vis_mask'] = torch.stack(patches_masks).cuda() #always on gpu during inference
        ret_dict['txt_in']   = ids.repeat(len(patches), 1).cuda() #always on gpu during inference
        ret_dict['txt_mask'] = ids_mask.repeat(len(patches), 1).cuda() #always on gpu during inference

        return ret_dict



@VILT_INTERFACE_REGISTRY.register()
def build_retrieval_interface(cfg, tokenizer, patch_size, prepare_patches_fn):
    return ViLTRetrievalInterface(max_patches=cfg.MAX_N_VISUAL,
                                 max_tokens=cfg.MAX_N_TOKENS,
                                 tokenizer=tokenizer,
                                 patch_size=patch_size,
                                 prepare_patches_fn=prepare_patches_fn)