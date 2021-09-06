
import random
from typing import List, TypeVar

import torch

Tensor = TypeVar('torch.tensor')



def masked_language_prep(tokens: List[str], tokenizer, probability_matrix: Tensor =None):

    ids    = tokenizer.convert_tokens_to_ids(tokens)
    ids    = torch.tensor(ids)
    labels = ids.clone().to(ids.device)

    special_tokens_mask = torch.tensor([0]*len(tokens))
    special_tokens_mask[0]  = 1
    special_tokens_mask[-1] = 1
    assert special_tokens_mask.shape[0] == ids.shape[0]

    if probability_matrix is None: #probability matrix is not None with WWM
        probability_matrix = torch.full(labels.shape, .15)
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100 #loss computed for masked tokens only

    #80% of time, replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, .8)).bool() & masked_indices
    ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    #10% of time, replace with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, .5)).bool() & masked_indices & ~indices_replaced
    random_words   = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    ids[indices_random] = random_words[indices_random]

    return ids, labels


def whole_word_mask(tokens: List[str], tokenizer, max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(tokens) * .15))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)
        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1.0 if i in covered_indexes else 0.0 for i in range(len(tokens))]
        return masked_language_prep(tokens, tokenizer, probability_matrix=torch.tensor(mask_labels))
