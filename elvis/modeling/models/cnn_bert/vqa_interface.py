import pdb
from collections import Counter
from typing import Dict, List, Text, TypeVar

import torch
from elvis.modeling.models.base import ModelDataInterface

from .dispatcher import CNN_BERT_INTERFACE_REGISTRY


class CNNBERTVQADataInterface(ModelDataInterface):
    def __init__(self, max_tokens, tokenizer, ans2id):
        self.max_tokens         = max_tokens
        self.tokenizer          = tokenizer
        self.ans2id             = ans2id

    def worker_fn(self, x):
        # dict_keys(['quest_id', 'img_id', 'quest', 'true_ans', 'ans_list', 'img'])
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

        assert ids.shape[0] == self.max_tokens+2
        ret_dict = {}
        ret_dict['img']       = x['img']
        ret_dict['ids']          = ids
        ret_dict['ids_mask']     = ids_mask
        ret_dict['gt_answers']   = ans_targets
        ret_dict['sample_id']    = x['quest_id']

        return ret_dict

    def collate_fn(self, batch):
        ret_dict = {}
        sample_ids   = [item['sample_id'] for item in batch]
        imgs         = [item['img'] for item in batch]
        ids          = [item['ids'] for item in batch]
        ids_mask     = [item['ids_mask'] for item in batch]
        answers      = [item['gt_answers'] for item in batch]

        ret_dict = {'samples_id': sample_ids,
                    'vis_in'    : torch.stack(imgs),
                    'vis_mask'  : None,
                    'txt_in'    : torch.stack(ids),
                    'txt_mask'  : torch.stack(ids_mask),
                    'gt_answers': torch.stack(answers)
                    }
        return ret_dict

    def worker_fn_eval(self, x):
        #dict_keys(['image_id', 'img', 'question', 'question_id'])
        tok_res      = self.tokenizer(x['question'], truncation=True, max_length=self.max_tokens+2, return_tensors='pt')
        ids          = tok_res['input_ids']
        ids_mask     = tok_res['attention_mask']

        ret_dict = {}
        ret_dict['question_id'] = x['question_id']
        ret_dict['image_id']    = x['image_id']
        ret_dict['vis_in']      = x['img'].unsqueeze(0)
        ret_dict['vis_mask']    = None #to make it compatible with the meta architecture
        ret_dict['txt_in']      = ids
        ret_dict['txt_mask']    = ids_mask

        return ret_dict

@CNN_BERT_INTERFACE_REGISTRY.register()
def build_vqa_interface(cfg, tokenizer, ans2id):
    return CNNBERTVQADataInterface(max_tokens=cfg.MAX_N_TOKENS,
                                   tokenizer=tokenizer,
                                   ans2id=ans2id)
