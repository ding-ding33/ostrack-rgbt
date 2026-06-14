import math
import logging
import pdb
from functools import partial, reduce
from collections import OrderedDict
from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_adapter


from ..layers.attn_adapt_blocks import CEABlock
from ..layers.dualstream_attn_blocks import DSBlock ## Dual Stream without adapter


from lib.models.layers.attn import Attention
from lib.models.layers.adapter import Bi_direct_adapter
from lib.models.layers.adapter import LightweightIQA_MLP, VIB_Feature_Purifier, CrossAttnRepair


_logger = logging.getLogger(__name__)



class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, adapter_type=None,
                 iqa_threshold=0.4, vib_reduction=8, vib_beta=0.001,
                 repair_num_layers=2, enable_modality_repair=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #self.patch_embed_adapter = embed_layer(
        #    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        self.pos_drop = nn.Dropout(p=drop_rate)

        
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """     #
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        depth = 12
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:  #ce_loc [3,6,9]
                ce_keep_ratio_i = ce_keep_ratio[ce_index]  #[1,1,1]
                ce_index += 1
            # if i ==9 or i ==10 or i ==11:
            #     blocks.append(
            #     CEABlock_Single(
            #             dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
            #             attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
            #             keep_ratio_search=ce_keep_ratio_i)
            #     )
            if i<20:
                blocks.append(
                CEABlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i)
                )
            else:
                blocks.append(
                    DSBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i)
                )
        

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        # -------- modality repair modules --------
        self.enable_modality_repair = enable_modality_repair
        self.iqa_threshold = iqa_threshold
        if self.enable_modality_repair:
            self.rgb_iqa_adapter = LightweightIQA_MLP(embed_dim, reduction=16, act_layer=act_layer)
            self.dte_iqa_adapter = LightweightIQA_MLP(embed_dim, reduction=16, act_layer=act_layer)
            self.rgb_vib_purifier = VIB_Feature_Purifier(embed_dim, reduction=vib_reduction, beta=vib_beta)
            self.dte_vib_purifier = VIB_Feature_Purifier(embed_dim, reduction=vib_reduction, beta=vib_beta)
            self.cross_attn_repair = CrossAttnRepair(embed_dim, num_heads=num_heads,
                                                      num_layers=repair_num_layers)

        self.init_weights(weight_init)

    def concat_same_sequence_templates(self,tensor):  #从一个序列中抽取双模板
        batch_size, seq_len, hidden_size = tensor.shape
        # 重新调整张量形状: [16, 2, 64, 768]
        reshaped = tensor.view(batch_size // 2, 2, seq_len, hidden_size)
        # 在第二个维度上拼接，最终形状: [16, 128, 768]
        concatenated = reshaped.reshape(batch_size // 2, seq_len * 2, hidden_size)
        return concatenated



    def _apply_vib_if_needed(self, tokens, iqa_score, purifier):
        """始终对所有样本施加 VIB 净化，由 ReZero alpha 控制生效程度。"""
        B = tokens.shape[0]
        out_list = []
        kl_total = torch.zeros((), device=tokens.device, dtype=tokens.dtype)
        for b in range(B):
            purged, kl = purifier(tokens[b:b + 1])
            out_list.append(purged)
            kl_total = kl_total + kl
        kl_total = kl_total / B
        return torch.cat(out_list, dim=0), kl_total

    def _apply_repair_if_needed(self, rgb_tokens, dte_tokens,
                                 rgb_z_score, dte_z_score,
                                 rgb_x_score, dte_x_score):
        """始终进行双向交叉注意力修复，由 ReZero alpha 控制生效程度。"""
        B = rgb_tokens.shape[0]
        device = rgb_tokens.device

        out_rgb = self.cross_attn_repair(rgb_tokens, dte_tokens)
        out_dte = self.cross_attn_repair(dte_tokens, rgb_tokens)

        repair_info = {
            "rgb_repaired": torch.ones(1, device=device),
            "dte_repaired": torch.ones(1, device=device),
        }
        return out_rgb, out_dte, repair_info

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False,dynamic_template=None,Test=None,template_masks=None,frame_id=None,last_tokens=None,init_tokens=None):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        num = len(z)
        if Test is None and num>=2:
            z1,z2 = z[0],z[1]
            x_rgb = x[:, :3, :, :]
            x_dte = x[:, 3:, :, :]

            z1_rgb = z1[:, :3, :, :]
            z1_dte = z1[:, 3:, :, :]
            z2_rgb = z2[:, :3, :, :]
            z2_dte = z2[:, 3:, :, :]
            z_list,zi_list = [],[]
            x = x_rgb
            xi = x_dte
            #修改为统一采用两个模板
            for i in range(B):
                z_list.append(torch.cat([z1_rgb[i].unsqueeze(0),z2_rgb[i].unsqueeze(0)],dim=0))
                zi_list.append(torch.cat([z1_dte[i].unsqueeze(0),z2_dte[i].unsqueeze(0)],dim=0))
            z_rgb = torch.cat(z_list,dim=0)
            z_dte = torch.cat(zi_list,dim=0)
        else:
        # rgb_img
            x_rgb = x[:, :3, :, :]
            z_rgb = z[:, :3, :, :]
            # depth thermal event images
            x_dte = x[:, 3:, :, :]
            z_dte = z[:, 3:, :, :]
        # overwrite x & z
        x, z = x_rgb, z_rgb
        xi, zi = x_dte, z_dte
        self.test = Test
        B, H, W = x.shape[0], x.shape[2], x.shape[3]        
            # x_list,xi_list = [],[]
            # for i in range(B):
            #     x_list.append(torch.cat([x_rgb[i].unsqueeze(0),x_rgb[i].unsqueeze(0)],dim=0))
            #     xi_list.append(torch.cat([x_dte[i].unsqueeze(0),x_dte[i].unsqueeze(0)],dim=0))
            # x_rgb = torch.cat(x_list,dim=0)
            # x_dte = torch.cat(xi_list,dim=0)
        # else:
        # # rgb_img
        #     x_rgb = x[:, :3, :, :]
        #     z_rgb = z[:, :3, :, :]
        #     # depth thermal event images
        #     x_dte = x[:, 3:, :, :]
        #     z_dte = z[:, 3:, :, :]
        #     x,xi = x_rgb,x_dte

        # x_rgb = x[:, :3, :, :]
        # z_rgb = z[:, :3, :, :]
        # # depth thermal event images
        # x_dte = x[:, 3:, :, :]
        # z_dte = z[:, 3:, :, :]
        # x,xi = x_rgb,x_dte
        # overwrite x & z

        self.test = Test
        # B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # if Test is not None:
        #     z1_rgb = self.patch_embed(z1_rgb)
        #     z2_rgb = self.patch_embed(z2_rgb)
        #     z3_rgb = self.patch_embed(z3_rgb)
        #     # z = self.patch_embed(z)
        #     x,x_t = self.patch_embed(x)

        #     z1_dte = self.patch_embed(z1_dte)
        #     z2_dte = self.patch_embed(z2_dte)
        #     z3_dte = self.patch_embed(z3_dte)
        #     xi,xi_t = self.patch_embed(xi)
        #     # zi = self.patch_embed(zi)
        # else:
        z = self.patch_embed(z)
        # z2_rgb = self.patch_embed(z2_rgb)
        # z3_rgb = self.patch_embed(z3_rgb)
        # z = self.patch_embed(z)
        x,x_t = self.patch_embed(x)

        zi = self.patch_embed(zi)
        # z2_dte = self.patch_embed(z2_dte)
        # z3_dte = self.patch_embed(z3_dte)
        xi,xi_t = self.patch_embed(xi)
        # zi = self.patch_embed(zi)


        if dynamic_template == None:
            pass
        else:
            self.dynamic_template = dynamic_template

###################################################################===========
        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x_t += self.pos_embed_z
        x += self.pos_embed_x

        xi_t += self.pos_embed_z
        zi += self.pos_embed_z
        xi += self.pos_embed_x
        

        # conatenate same sequence templates for training   
        # if Test is None:
        z = self.concat_same_sequence_templates(z)
        zi = self.concat_same_sequence_templates(zi)
        if Test is not None:
            x = x[0].unsqueeze(0)
            xi = xi[0].unsqueeze(0)

        # -------- modality repair: IQA → VIB → cross-attention --------
        repair_info = None
        vib_kl_info = None
        iqa_scores = None
        if self.enable_modality_repair:
            z_iqa = self.rgb_iqa_adapter(z)
            x_iqa = self.rgb_iqa_adapter(x)
            zi_iqa = self.dte_iqa_adapter(zi)
            xi_iqa = self.dte_iqa_adapter(xi)

            z = z * (2.0 * z_iqa).unsqueeze(1)
            x = x * (2.0 * x_iqa).unsqueeze(1)
            zi = zi * (2.0 * zi_iqa).unsqueeze(1)
            xi = xi * (2.0 * xi_iqa).unsqueeze(1)

            z, kl_z_rgb = self._apply_vib_if_needed(z, z_iqa, self.rgb_vib_purifier)
            x, kl_x_rgb = self._apply_vib_if_needed(x, x_iqa, self.rgb_vib_purifier)
            zi, kl_zi_dte = self._apply_vib_if_needed(zi, zi_iqa, self.dte_vib_purifier)
            xi, kl_xi_dte = self._apply_vib_if_needed(xi, xi_iqa, self.dte_vib_purifier)

            lens_z_rep = z.shape[1]
            lens_x_rep = x.shape[1]

            rgb_tokens = torch.cat([z, x], dim=1)
            dte_tokens = torch.cat([zi, xi], dim=1)

            rgb_tokens, dte_tokens, repair_info = self._apply_repair_if_needed(
                rgb_tokens, dte_tokens,
                z_iqa, zi_iqa, x_iqa, xi_iqa)

            z = rgb_tokens[:, :lens_z_rep]
            x = rgb_tokens[:, lens_z_rep:]
            zi = dte_tokens[:, :lens_z_rep]
            xi = dte_tokens[:, lens_z_rep:]

            vib_kl_info = {
                "rgb": kl_z_rgb + kl_x_rgb,
                "dte": kl_zi_dte + kl_xi_dte,
                "total": kl_z_rgb + kl_x_rgb + kl_zi_dte + kl_xi_dte,
            }
            iqa_scores = {
                "rgb": {
                    "template": z_iqa,
                    "search": x_iqa,
                },
                "dte": {
                    "template": zi_iqa,
                    "search": xi_iqa,
                },
            }

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed
            xi += self.search_segment_pos_embed      #//////////////////////////////////////////////////////////////////
            zi += self.template_segment_pos_embed
        #print(x.shape) #[Batch size, 256, 768]
        #z [bs,64,768]
        x = combine_tokens(z, x, mode=self.cat_mode)  ##[Batch size, 320, 768]
        #print("after cat",x.shape)

        xi = combine_tokens(zi, xi, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)
            xi = torch.cat([cls_tokens, xi], dim=1)

        x = self.pos_drop(x)
        xi = self.pos_drop(xi)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        global_index_ti = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_ti = global_index_ti.repeat(B, 1)

        global_index_si = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_si = global_index_si.repeat(B, 1)


        removed_indexes_s = []
        removed_indexes_si = []
        #用于统计两个模态间ce的差值
        # diff_sum = 0
        x_list = []
        test_tokens = True
        removed_flag = False

        for i, blk in enumerate(self.blocks[:12]):
            # if i ==9 or i ==10 or i ==11:
            x, global_index_t, global_index_s, removed_index_s, attn, \
            xi, global_index_ti, global_index_si, removed_index_si, attn_i = \
                blk(x, xi, global_index_t, global_index_ti, global_index_s, global_index_si, mask_x, ce_template_mask,
                    ce_keep_rate,dynamic_template,Test=Test)
            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)
                removed_indexes_si.append(removed_index_si)
            # else:
            #     x, global_index_t, global_index_s, removed_index_s, attn, \
            #     xi, global_index_ti, global_index_si, removed_index_si, attn_i = \
            #         blk(x, xi, global_index_t, global_index_ti, global_index_s, global_index_si, mask_x, ce_template_mask,
            #             ce_keep_rate,dynamic_template)
            #     if self.ce_loc is not None and i in self.ce_loc:
            #         removed_indexes_s.append(removed_index_s)
            #         removed_indexes_si.append(removed_index_si)

        x = self.norm(x)
        xi = self.norm(xi)
   

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]
        lens_xi_new = global_index_si.shape[1]
        lens_zi_new = global_index_ti.shape[1]

        lens_s = 256

        z = x[:, :-lens_s]
        x = x[:, -lens_s:]
        zi = xi[:, :-lens_s]
        xi = xi[:, -lens_s:]

 

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
        
        if removed_indexes_si and removed_indexes_si[0] is not None:
            removed_indexes_cat_i = torch.cat(removed_indexes_si, dim=1)

            pruned_lens_xi = lens_x - lens_xi_new                                ########################
            pad_xi = torch.zeros([B, pruned_lens_xi, xi.shape[2]], device=xi.device)
            xi = torch.cat([xi, pad_xi], dim=1)
            index_all = torch.cat([global_index_si, removed_indexes_cat_i], dim=1)
            # recover original token order
            C = xi.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            xi = torch.zeros_like(xi).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=xi)
        
        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        xi = recover_tokens(xi, lens_zi_new, lens_x, mode=self.cat_mode)



        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)
        xi = torch.cat([zi, xi], dim=1)
        #x = torch.cat([x, xi], dim=0)


        x = x + xi
        B,L,C = x.shape

        dynamic_tokens = None
        aux_dict = {
            "attn": attn,
            "t_weights": None,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "vib_kl_info": vib_kl_info,
            "iqa_scores": iqa_scores,
            "repair_info": repair_info,
        }
        return x, aux_dict, dynamic_tokens
    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False,dynamic_template=None,Test=None,template_masks=None,frame_id=None):
        dynamic_template_ = dynamic_template

        x, aux_dict,dynamic_template= self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,dynamic_template=dynamic_template_,Test=Test,template_masks=template_masks,frame_id=frame_id,last_tokens=dynamic_template_)
        if dynamic_template is not None:
            dynamic_template_save = dynamic_template.detach()

        else:
            dynamic_template_save = None

        return x, aux_dict,dynamic_template_save


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_adapter(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_adapter(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
