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


class ViLT(nn.Module):
    """Vision-and-Language Transformer

    A PyTorch implementation of `ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision`  -
        https://arxiv.org/abs/2102.03334
    """
    def __init__(self, vit_model, max_n_patches, max_n_words, pretrained_tokenizer, pretrained_word_embeddings=None, vit_pretrained=False):
        super(ViLT, self).__init__()
        self.max_n_patches = max_n_patches+1 #take into account [IMG] embedding
        fn = 'timm.models.{}'
        vit = eval(fn.format(vit_model))(pretrained=vit_pretrained)
        #initialize vit transformer, linear projection and embeddings from pretrained
        self._create_vilt_from_vit(vit)
        del vit
        self.max_n_words = max_n_words+2 #take into account [CLS] and [SEP]
        self.tokenizer   = BertTokenizer.from_pretrained(pretrained_tokenizer)
        if pretrained_word_embeddings:
            self._initialize_pretrained_embeddings(pretrained_word_embeddings)
        else:
            self.w_embed     = nn.Embedding(num_embeddings=self.tokenizer.vocab_size, embedding_dim=768, padding_idx=0, sparse=False) #sparse not supported with AdamW
            self.t_pos_embed = nn.Parameter(torch.zeros(self.max_n_words, 768)) 
            torch.nn.init.xavier_uniform_(self.t_pos_embed)
        
        self.v_mod_emb   = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.t_mod_emb   = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.v_emb_norm  = nn.LayerNorm(self.embed_dim)
        self.t_emb_norm  = nn.LayerNorm(self.embed_dim)
        torch.nn.init.xavier_uniform_(self.v_mod_emb)
        torch.nn.init.xavier_uniform_(self.t_mod_emb)
    
    def _create_vilt_from_vit(self, vit):
        self.patch_size         = vit.patch_embed._modules['proj'].kernel_size[0]
        self.flatten_patch_size = self.patch_size**2*3 #p^2 * n_channels
        self.embed_dim          = vit.embed_dim
        self.v_cls              = vit.cls_token
        self.patch_proj         = nn.Linear(in_features=self.flatten_patch_size, out_features=vit.embed_dim)
        self.patch_proj.weight  = nn.Parameter(vit.patch_embed.proj.weight.view(vit.embed_dim, -1))
        #here get positional embeddings from interpolation
        #vit.pos_embed.shape = torch.Size([1, 50, 768])
        self.v_pos_embed        = nn.Parameter(F.interpolate(vit.pos_embed.unsqueeze(1), size=(self.max_n_patches, vit.embed_dim)).squeeze(1))
        self.pos_drop           = vit.pos_drop
        self.transformer        = vit.blocks
        self.norm               = vit.norm

    def _initialize_pretrained_embeddings(self, pretrained):
        #initialize word and position embeddings from BERT
        bert             = BertModel.from_pretrained(pretrained)
        self.w_embed     = bert.embeddings.word_embeddings
        self.t_pos_embed = nn.Parameter(bert.embeddings.position_embeddings.weight[:self.max_n_words])
        del bert

    def forward(self, vis_in: Tensor, txt_in: Tensor, vis_mask: Tensor, txt_mask: Tensor):
        assert vis_in.shape[0] == txt_in.shape[0], 'Unconsistent batch size'
        #assert vis_in.shape[1] == self.max_n_patches-1 #take into account [IMG]
        B     = vis_in.shape[0]
        T_LEN = txt_in.shape[1]

        v_emb = self.patch_proj(vis_in)
        t_emb = self.w_embed(txt_in)

        #textual input already have [CLS]
        v_cls = self.v_cls.expand(B, -1, -1)
        v_emb = torch.cat((v_cls, v_emb), dim=1)
        V_LEN = v_emb.shape[1]

        v_emb = self.v_emb_norm(v_emb + self.v_pos_embed[:, :V_LEN] + self.v_mod_emb)
        t_emb = self.t_emb_norm(t_emb + self.t_pos_embed[:T_LEN] + self.t_mod_emb)
        v_emb = self.pos_drop(v_emb)
        t_emb = self.pos_drop(t_emb)

        #build transformer input sequence and attention mask
        x = torch.cat((t_emb, v_emb), dim=1)
        attn_mask = torch.cat((
                                txt_mask,
                                torch.ones(B, 1).to(vis_mask.device), #vis_cls attention mask
                                vis_mask
                               ),
                               dim=-1)
        assert attn_mask.shape[-1] == x.shape[1], 'Wrong dimension for the attention mask'
        attn_mask = attn_mask.unsqueeze(-1) * attn_mask.unsqueeze(-2)
        for blk in self.transformer:
            x = self.vilt_block_forward(blk, x, attn_mask)
        #x = self.pre_logits(x)
        #t_pool = self.norm(x)[:, 0]
        #v_pool = self.norm(x)[:, T_LEN]
        return x
    
    def vilt_block_forward(self, blk, x, attn_mask):
        """This method extend the original ViT forward from timm library to incorporate masking operations
        Args:
            blk (nn.Module): ViT transformer block
            x (Tensor): input Tensor
        """
        x = x + blk.drop_path(self.masked_attn(blk.attn, blk.norm1(x), attn_mask))
        x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
        return x

    def masked_attn(self, attn_blk, x, attn_mask):
        B, N, C = x.shape
        qkv     = attn_blk.qkv(x).reshape(B, N, 3, attn_blk.num_heads, C // attn_blk.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn           = (q @ k.transpose(-2, -1)) * attn_blk.scale
        _MASKING_VALUE = -1e+30 if attn.dtype == torch.float32 or attn.dtype == torch.float64 else -1e+4
        attn           = attn.masked_fill(attn_mask[:, None, :, :]==0, value=_MASKING_VALUE)

        attn = attn.softmax(dim=-1)
        attn = attn_blk.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_blk.proj(x)
        x = attn_blk.proj_drop(x)
        return x

    @staticmethod
    def prepare_patches(img: Tensor, patch_size:int):
        #this is the same of using enops.rearrange(rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = 32, p2 = 32))
        # it is not clear wheter this is the correct implementation or
        #  enops.rearrange(rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 32, p2 = 32))
        x       = img.unsqueeze(0) if img.dim() == 3 else img
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size).permute(0, 2, 1)
        return patches.squeeze(0)

    def pad_and_mask_patches(self, v: List[Tensor], max_size: int =-1):
        """Method to pad and mask the visual and textual inputs accordingly to the model needs

        Args:
            v (Tensor): visual input
            t (Tensor): textual input
            max_v_size (int, optional): visual input size after padding. Set to -1 for dynamically reshaping. Defaults to -1.
            max_t_size (int, optional): textual input size after padding. Set to -1 for dynamically reshaping. Defaults to -1.
        """
        vis_lens    = list(map(lambda item: item.shape[0], v))
        max_len     = max(vis_lens) if max_size == -1 else max_size
        vis_in      = torch.zeros(len(v), max_len, self.flatten_patch_size)
        vis_mask    = torch.zeros(len(v), max_len, dtype=torch.int64)
        for idx, curr_len in enumerate(vis_lens):
            vis_in[idx][:curr_len] = v[idx]
            vis_mask[idx][:curr_len] = 1
        return vis_in, vis_mask

    def pad_and_mask_text(self, t: List[Tensor], max_size: int = -1):
        if max_size == -1:
            res = self.tokenizer(t, padding='longest', return_tensors='pt')
        else:
            res = self.tokenizer(t, padding='max_length', max_length=max_size, return_tensors='pt')
        return res['input_ids'], res['attention_mask']
        


@NET_REGISTRY.register()
def build_vilt_model(cfg, get_interface=None, **kwargs):
    model = ViLT(vit_model=cfg.NET.TYPE,
                 max_n_patches=cfg.MAX_N_VISUAL,
                 max_n_words=cfg.MAX_N_TOKENS,
                 pretrained_tokenizer=cfg.NET.TOKENIZER,
                 pretrained_word_embeddings=cfg.NET.PRETRAINED_EMBEDDINGS if cfg.NET.has_attr('PRETRAINED_EMBEDDINGS') else None,
                 vit_pretrained=cfg.NET.PRETRAINED)
    if get_interface is not None:
        args_dict = {'cfg': cfg, 'tokenizer': model.tokenizer, 'patch_size': model.patch_size, 'prepare_patches_fn': model.prepare_patches}
        args_dict.update(kwargs)
        interface = build_data_interface(get_interface, **args_dict)
        return model, interface
    return model


