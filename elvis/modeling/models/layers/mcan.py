import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F

from .fully_connected import MLP
from .transformer import TRMEncoderLayer, TRMMHAttention

__all__ = ['MCANEncoderLayer',
           'MCANDecoderLayer',
           'MCANOutputLayer']


class MCANEncoderLayer(TRMEncoderLayer):
    """Encoder layer for MCAN. It is a wrapper around the typical Transformer Encoder Layer
    """
    def __init__(self, n_heads, d_model, dropout_p=0.):
        super(MCANEncoderLayer, self).__init__(n_heads, d_model, dropout_p)

    def forward(self, lang_feats, lang_mask):
        return super(MCANEncoderLayer, self).forward(inputs=lang_feats, attn_mask=lang_mask)


class MCANDecoderLayer(nn.Module):
    """Implementation of the SGA layer of MCAN
    """

    def __init__(self, n_heads, d_model, dropout_p=0.):
        super(MCANDecoderLayer, self).__init__()
        self.sa     = TRMMHAttention(n_heads, d_model, dropout_p)
        self.norm_1 = nn.LayerNorm(d_model)
        self.drop_1 = nn.Dropout(p=dropout_p)

        self.ga     = TRMMHAttention(n_heads, d_model, dropout_p)
        self.norm_2 = nn.LayerNorm(d_model)
        self.drop_2 = nn.Dropout(p=dropout_p)

        self.ffn = MLP(in_features=d_model,
                        hidden_dim=d_model//2,
                        out_features=d_model,
                        dropout_p=dropout_p,
                        use_relu=True)
        self.norm_3 = nn.LayerNorm(d_model)
        self.drop_3 = nn.Dropout(p=dropout_p)

    def forward(self, vis_feats, lang_feats, vis_mask, lang_mask):
        out_sa, attn_scores_sa = self.sa(vis_feats, vis_feats, vis_mask, vis_mask)
        out_sa = self.norm_1(vis_feats + self.drop_1(out_sa))

        out_ga, attn_scores_ga = self.ga(vis_feats, lang_feats, vis_mask, lang_mask)
        out_ga = self.norm_2(out_sa + self.drop_2(out_ga))

        out_ffn = self.norm_3(out_ga + self.drop_3(self.ffn(out_ga)))

        return out_ffn, attn_scores_ga, attn_scores_sa


class MCANOutputLayer(nn.Module):
    def __init__(self, d_model, d_out, dropout_p=0.):
        super(MCANOutputLayer, self).__init__()
        #language attention reduce
        self.l_attn_red = MCANAttnReduce(in_features=d_model,
                                        hidden_dim=d_model//2,
                                        dropout_p=dropout_p)
        #vision attention reduce
        self.v_attn_red = MCANAttnReduce(in_features=d_model,
                                        hidden_dim=d_model//2,
                                        dropout_p=dropout_p)

        self.l_out = nn.Linear(in_features=d_model,
                                out_features=d_out,
                                bias=False)
        self.v_out = nn.Linear(in_features=d_model,
                                out_features=d_out,
                                bias=False)
        self.norm = nn.LayerNorm(d_out)
    
    def forward(self, v_feats, l_feats, v_mask, l_mask):
        l_red = self.l_attn_red(l_feats, l_mask)
        v_red = self.v_attn_red(v_feats, v_mask)
        return self.norm(self.l_out(l_red) + self.v_out(v_red))


class MCANAttnReduce(nn.Module):
    #TODO add mask for attention
    def __init__(self, in_features, hidden_dim, dropout_p=0.):
        super(MCANAttnReduce, self).__init__()
        self.mlp = MLP(in_features=in_features,
                        hidden_dim=hidden_dim,
                        out_features=1,
                        dropout_p=dropout_p,
                        use_relu=True)
    
    def forward(self, inputs, in_mask):
        #output size: [BxIN_LEN]
        attn_logits = self.mlp(inputs).squeeze(-1)
        _MASKING_VALUE = -1e+30 if attn_logits.dtype == torch.float32 else -1e+4
        masked_attn_logits = attn_logits.masked_fill(in_mask==0, value=_MASKING_VALUE)
        attn_scores = F.softmax(masked_attn_logits, dim=-1)
        return torch.bmm(attn_scores.unsqueeze(-1).transpose(1, 2), inputs).squeeze(1)
