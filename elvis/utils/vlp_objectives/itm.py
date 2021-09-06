
import pdb
import random
from typing import Dict

import torch


def itm_preparation_fn(pos_pair: Dict, neg_pair: Dict, tokenizer, max_len: int):
    ret_pair  = pos_pair
    #ret_pair  = {'img': pos_pair['img'], 'text': pos_pair['caption'], 'id': pos_pair['id'], 'dataset': pos_pair['dataset']}
    negpair_p = random.uniform(0, 1)
    if negpair_p > .5:
        if random.uniform(0, 1) > .5:
            ret_pair['img']     = neg_pair['img'] #exchange
        else:
            ret_pair['text']    = neg_pair['text'] #exchange
    ids       = tokenizer(ret_pair['text'], return_tensors='pt')['input_ids'].squeeze(0)
    ids       = ids[:max_len]
    target    = torch.full(ids.shape, -100)
    target[0] = 0 if negpair_p > .5 else 1
    return ret_pair, ids, target
