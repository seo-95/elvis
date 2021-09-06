import math
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F

from .fully_connected import MLP


class TRMMHAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout_p=0.):
        super(TRMMHAttention, self).__init__()
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_k        = d_model // n_heads
        self.dropout_p  = dropout_p
        #instantiate single Q, K, and V weights and split them in multiple heads during forward (better parallelization on GPU)
        # this is better than creating n_heads weights matrices and run them in a loop
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        #self.qkv = nn.Linear(3*d_model, d_model) #?

        self.dropout    = nn.Dropout(p=dropout_p)
        self.out        = nn.Linear(d_model, d_model)

    def forward(self, in_1, in_2, attn_mask_1, attn_mask_2):
        assert in_1.shape[0] == in_2.shape[0], 'Uncompatible batch sizes'
        b_size = in_1.shape[0]

        # split the result of linear operations in n_heads heads (b_size, seq_len, n_heads, d_k)
        q = self.Q(in_1).view(b_size, -1, self.n_heads, self.d_k)
        k = self.K(in_2).view(b_size, -1, self.n_heads, self.d_k)
        v = self.V(in_2).view(b_size, -1, self.n_heads, self.d_k)

        #transpose to get dimension (b_size, n_heads, seq_len, d_k)
        # this because torch.matmul performs matrix multiplication between the last two dimensions (all the other are considered batch)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        #build attn mask by projecting the set_1 mask on the columns and set_2 mask on the rows
        attn_mask           = attn_mask_1.unsqueeze(-1) * attn_mask_2.unsqueeze(-2)
        outs, attn_scores   = self.scaledDotProductAttention(q, k, v, attn_mask)
        out_concat          = outs.transpose(1, 2).contiguous().view(b_size, -1, self.d_model)
        #attn_scores         = attn_scores.transpose(1, 2)

        return self.out(out_concat), attn_scores

    #? detach it from the multi-head attn class? In case there is the need to pass an nn.Dropout() as parameter
    def scaledDotProductAttention(self, query, key, value, attn_mask):
        attn_logits = torch.matmul(query, torch.transpose(key, -2, -1))/ math.sqrt(key.shape[-1])
        #mask padding and future words with a great negative value.
        # DO NOT USE '-INF' BECAUSE IT WILL GENERATE NaN AFTER SOFTMAX FOR PADDING ROWS (filled with all 0's)
        #take into account mixed precision and avoid overflow (-1e+4 can be represented on 16 bits)
        _MASKING_VALUE = -1e+30 if attn_logits.dtype == torch.float32 else -1e+4
        if attn_logits.dim() == 4:
            #in case of multi-head attentions there is an additional dimension given by the n of heads
            masked_attn_logits = attn_logits.masked_fill(attn_mask[:, None, :, :]==0, value=_MASKING_VALUE)
        else:
            masked_attn_logits = attn_logits.masked_fill(attn_mask==0, value=_MASKING_VALUE)

        attn_scores = F.softmax(masked_attn_logits, dim=-1)
        attn_scores = self.dropout(attn_scores)
        out = torch.matmul(attn_scores, value)
        return out, attn_scores


class TRMEncoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, dropout_p=0.):
        super(TRMEncoderLayer, self).__init__()
        self.mhead_attn     = TRMMHAttention(n_heads=n_heads,
                                            d_model=d_model,
                                            dropout_p=dropout_p)
        self.dropout_1      = nn.Dropout(p=dropout_p)
        self.norm_1         = nn.LayerNorm(d_model)
        self.mlp            = MLP(in_features=d_model,
                                hidden_dim=d_model//2,
                                out_features=d_model,
                                dropout_p=dropout_p,
                                use_relu=True)
        self.dropout_2      = nn.Dropout(p=dropout_p)
        self.norm_2         = nn.LayerNorm(d_model)

    def forward(self, inputs, attn_mask):
        mhead_out, attn_scores  = self.mhead_attn(inputs, inputs, attn_mask, attn_mask)
        mid_out                 = self.norm_1(self.dropout_1(mhead_out) + inputs)
        out                     = self.norm_2(self.dropout_2(self.mlp(mid_out)) + mid_out)
        return out, attn_scores


class TRMDecoderLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_ff, dropout_p=0.):
        super(TRMDecoderLayer, self).__init__()
        self.mhead_self_attn    = TRMMHAttention(n_heads=n_heads,
                                                d_model=d_model,
                                                dropout_p=dropout_p)
        self.dropout_1          = nn.Dropout(p=dropout_p)
        self.norm_1             = nn.LayerNorm(d_model)

        self.mhead_enc_attn     = TRMMHAttention(n_heads=n_heads,
                                                d_model=d_model,
                                                dropout_p=dropout_p)
        self.dropout_2          = nn.Dropout(p=dropout_p)
        self.norm_2             = nn.LayerNorm(d_model)

        self.mlp                = MLP(in_features=d_ff,
                                    hidden_dim=d_ff//2,
                                    out_features=d_ff,
                                    dropout_p=dropout_p,
                                    use_relu=True)
        self.dropout_3          = nn.Dropout(p=dropout_p)
        self.norm_3             = nn.LayerNorm(d_model)

    def forward(self, inputs, enc_outs, inputs_mask, enc_mask):
        s_mhead_out, s_attn_scores  = self.mhead_self_attn(inputs, inputs, attn_mask=inputs_mask)
        mid_out_1                   = self.norm_1(self.dropout_1(s_mhead_out) + inputs)

        joint_mask = None #TODO compute joint mask between inputs and enc_outs
        e_mhead_out, e_attn_scores  = self.mhead_enc_attn(inputs, enc_outs, attn_mask=joint_mask)
        mid_out_2                   = self.norm_1(self.dropout_1(e_mhead_out) + mid_out_1)

        out                     = self.norm_2(self.dropout_2(self.mlp(mid_out_2)) + mid_out_2)
        return out, s_attn_scores, e_attn_scores
