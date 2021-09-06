import pdb
from typing import Dict, List, Text, TypeVar

import torch
from elvis.modeling.models.base import ModelDataInterface
from elvis.utils.vlp_objectives import (itm_preparation_fn,
                                        masked_language_prep, whole_word_mask)

from .dispatcher import VILT_INTERFACE_REGISTRY


class ViLTPretrainDataInterface(ModelDataInterface):

    def __init__(self, max_patches, max_tokens, tokenizer, tasks_dict, patch_size, prepare_patches_fn):
        self.prepare_patches_fn = prepare_patches_fn
        self.max_patches        = max_patches
        self.max_tokens         = max_tokens
        self.tokenizer          = tokenizer
        self.patch_size         = patch_size
        self.tasks_dict         = tasks_dict
        self.lm_masking_fn      = whole_word_mask if self.tasks_dict['MLM']['WWM'] else masked_language_prep
        self.itm_prepare_fn     = itm_preparation_fn

    def worker_fn(self, x, neg_sample):
        """Pad input, build mask and language model targets.

        Args:
            x ([type]): [description]

        Returns:
            [Dict]: [description]
        """
        #only one among ITM and MLM are active for a given sample. Put the targets for the inactive one to -100
        #task_p = random.uniform(0, 1)
        #task   = 'MLM' if task_p > .5 else 'ITM'
        task = 'MLM' if neg_sample is None else 'ITM'
        if task == 'ITM':
            x, ids, itm_targets = self.itm_prepare_fn(x, neg_sample, self.tokenizer, max_len=self.max_tokens)
            lm_targets = torch.full(itm_targets.shape, -100)
        else:
            tokens          = self.tokenizer.tokenize(x['text'])
            tokens          = tokens[:self.max_tokens] #take into account [CLS] and [SEP]
            tokens.insert(0, '[CLS]')
            tokens.append('[SEP]')
            ids, lm_targets = self.lm_masking_fn(tokens, self.tokenizer)
            itm_targets     = torch.full(lm_targets.shape, -100)

        #attention mask and padding for text
        ids_mask       = torch.ones(ids.shape)
        to_pad         = self.max_tokens+2 - ids.shape[0] 
        if ids.shape[0] < self.max_tokens+2:
            padding     = torch.full([to_pad], self.tokenizer.pad_token_id)
            ids         = torch.cat((ids, padding), dim=0)
            padding     = torch.full([to_pad], -100)
            itm_targets = torch.cat((itm_targets, padding), dim=0)
            lm_targets  = torch.cat((lm_targets, padding), dim=0)
            padding     = torch.zeros(to_pad)
            ids_mask    = torch.cat((ids_mask, padding), dim=0) 

        #attention mask and padding for image
        patches      = self.prepare_patches_fn(x['img'], self.patch_size)
        patches      = patches[:self.max_patches]
        patches_mask = torch.ones(patches.shape[0])
        to_pad       = self.max_patches - patches.shape[0]
        if to_pad > 0:
            padding      = torch.zeros(to_pad, patches.shape[-1])
            patches      = torch.cat((patches, padding), dim=0)
            padding      = torch.zeros(to_pad)
            patches_mask = torch.cat((patches_mask, padding), dim=0)

        assert patches.shape[0] == self.max_patches and ids.shape[0] == self.max_tokens+2

        ret_dict = {}
        ret_dict['patches']        = patches
        ret_dict['patches_mask']   = patches_mask
        ret_dict['ids']            = ids
        ret_dict['ids_mask']       = ids_mask
        ret_dict['lm_targets']     = lm_targets
        ret_dict['itm_targets']    = itm_targets
        ret_dict['sample_id']      = x['id']
        ret_dict['task']           = task

        return ret_dict

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """This method is called after all the samples for the current batch were fetched.
            It is used to correctly prepare the batch to be fed to the model.
        """
        ret_dict = {}
        sample_ids   = [item['sample_id'] for item in batch]
        patches      = [item['patches'] for item in batch]
        patches_mask = [item['patches_mask'] for item in batch]
        ids          = [item['ids'] for item in batch]
        ids_mask     = [item['ids_mask'] for item in batch]
        lm_targets   = [item['lm_targets'] for item in batch]
        itm_targets  = [item['itm_targets'] for item in batch]
        tasks        = [item['task'] for item in batch]

        ret_dict = {'samples_id' : sample_ids,
                    'vis_in'     : torch.stack(patches),
                    'vis_mask'   : torch.stack(patches_mask),
                    'txt_in'     : torch.stack(ids),
                    'txt_mask'   : torch.stack(ids_mask),
                    'lm_targets' : torch.stack(lm_targets),
                    'itm_targets': torch.stack(itm_targets),
                    'tasks'      : tasks,
                    'ot_switch'  : torch.tensor([int(t == 'ITM') for t in tasks], dtype=torch.long), #? not used
                    }
        return ret_dict
    """
    @staticmethod
    def _compute_ot_scatter(txt_lens, max_txt_len, joint_len):
        pdb.set_trace()
        ot_scatter = torch.arange(0, joint_len, dtype=torch.long
                                ).unsqueeze(0).repeat(len(txt_lens), 1)
        for i, tl in enumerate(txt_lens):
            max_ind = max_txt_len + (joint_len-tl)
            ot_scatter.data[i, tl:] = torch.arange(max_txt_len, max_ind,
                                                dtype=torch.long).data
        return ot_scatter
    """



@VILT_INTERFACE_REGISTRY.register()
def build_vlp_interface(cfg, tokenizer, patch_size, prepare_patches_fn):
    return ViLTPretrainDataInterface(max_patches=cfg.MAX_N_VISUAL,
                                     max_tokens=cfg.MAX_N_TOKENS,
                                     tokenizer=tokenizer,
                                     tasks_dict=cfg.TASKS.get_as_dict(),
                                     patch_size=patch_size,
                                     prepare_patches_fn=prepare_patches_fn)

