import pdb
from typing import Dict, List, TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from elvis.modeling.models.build import NET_REGISTRY
from transformers import BertModel, BertTokenizer

from .dispatcher import build_data_interface

Tensor  = TypeVar('torch.tensor')

class VIT_BERT(nn.Module):
    """VIT followed by Transformer
    """

    def __init__(self, vit_model, max_n_patches, max_n_tokens, pretrained_bert, freeze_vit=False):
        super(VIT_BERT, self).__init__()
        self.max_n_patches = max_n_patches
        self.max_n_tokens = max_n_tokens

        #resnet
        fn          = 'timm.models.{}'
        self.vit = eval(fn.format(vit_model))(pretrained=True)
        del self.vit.head
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        #bert
        self.tokenizer       = BertTokenizer.from_pretrained(pretrained_bert)
        bert                 = BertModel.from_pretrained(pretrained_bert)
        self.bert_encoder    = bert.encoder
        self.w_embeddings    = bert.embeddings.word_embeddings
        self.pos_embeddings  = nn.Parameter(bert.embeddings.position_embeddings.weight)
        self.embeddings_norm = bert.embeddings.LayerNorm
        self.embeddings_drop = bert.embeddings.dropout
        
        self.embed_dim = bert.encoder.layer[0].output.dense.out_features
        self.v_mod_emb = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.t_mod_emb = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.v_emb_size   = self.vit.blocks[-1].mlp.fc2.out_features
        self.v_projection = nn.Linear(in_features=self.v_emb_size, out_features=self.embed_dim)
    

    def forward(self, vis_in: Tensor, txt_in: Tensor, txt_mask: Tensor, vis_mask=None):
        #vis_mask is set to None in order to keep compatibility with meta architecture
        B_SIZE = vis_in.shape[0]

        #compute visual features
        v_emb    = self.vit_forward(vis_in)
        v_emb    = self.v_projection(v_emb) #768 -> 768
        V_LEN    = v_emb.shape[1]
        vis_mask = torch.ones(B_SIZE, V_LEN).to(txt_in.device)

        #prepare text embeddings
        t_emb = self.w_embeddings(txt_in)
        T_LEN = txt_in.shape[1]

        #prepare positional and modal-aware embeddings
        v_emb = self.embeddings_norm(v_emb + self.pos_embeddings[:V_LEN] + self.v_mod_emb)
        t_emb = self.embeddings_norm(t_emb + self.pos_embeddings[:T_LEN] + self.t_mod_emb)
        v_emb = self.embeddings_drop(v_emb)
        t_emb = self.embeddings_drop(t_emb)

        #build transformer input sequence and attention mask
        x         = torch.cat((t_emb, v_emb), dim=1)
        attn_mask = torch.cat((txt_mask, vis_mask), dim=-1)
        #attention mask for encoder has to be broadcastable
        out       = self.bert_encoder(x, attention_mask=attn_mask[:, None, None, :])
        out       = out['last_hidden_state']
        return out


    def vit_forward(self, vis_in):
        x         = self.vit.patch_embed(vis_in)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x         = torch.cat((cls_token, x), dim=1)
        x         = self.vit.pos_drop(x + self.vit.pos_embed)
        for idx in range(len(self.vit.blocks)):
            x = self.vit.blocks[idx](x)
        x         = self.vit.norm(x)
        return x

@NET_REGISTRY.register()
def build_vit_bert_model(cfg, get_interface=None, **kwargs):
    model =  VIT_BERT(vit_model=cfg.NET.VIT_MODEL,
                      max_n_patches=cfg.MAX_N_VISUAL,
                      max_n_tokens=cfg.MAX_N_TOKENS,
                      pretrained_bert=cfg.NET.PRETRAINED_BERT,
                      freeze_vit=cfg.NET.FREEZE_VIT)

    if get_interface is not None:
        args_dict = {'cfg': cfg, 'tokenizer': model.tokenizer}
        args_dict.update(kwargs)
        interface = build_data_interface(get_interface, **args_dict)
        return model, interface
