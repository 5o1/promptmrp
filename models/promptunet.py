"""
This file contains one implementation of the PromptMR+ model. default is v2.
"""
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from mri_utils import ifft2c, complex_mul, rss, complex_abs, rss_complex, sens_expand, sens_reduce
from .utils import KspaceACSExtractor, conv, CAB, DownBlock, UpBlock, SkipBlock, PromptBlock
from .modules import rearrange


def checknan(tensor, flag):
    if not torch.isfinite(tensor).all():
        raise ValueError(f'Tensor NaN in flag{flag}')

class PromptUnet(nn.Module):
    def __init__(self,
                in_chans: int,
                out_chans: int,
                n_feat0: int,
                feature_dim: List[int],
                prompt_dim: List[int],
                len_prompt: List[int],
                prompt_size: List[int],
                n_enc_cab: List[int],
                n_dec_cab: List[int],
                n_skip_cab: List[int],
                n_bottleneck_cab: int,
                kernel_size=3,
                reduction=4,
                bias=False,
                no_use_ca=False,
                learnable_prompt=False,
                adaptive_input=False,
                n_buffer=0,
                n_history=0,
                 ):
        super().__init__()
        act=nn.PReLU()

        self.feature_dim = feature_dim
        self.n_history = n_history
        self.n_buffer = n_buffer if adaptive_input else 0

        in_chans = in_chans * (1+self.n_buffer) if adaptive_input else in_chans
        out_chans = out_chans * (1+self.n_buffer) if adaptive_input else in_chans

        # Feature extraction
        self.feat_extract = conv(in_chans, n_feat0, kernel_size, bias=bias).rearrange("b t s c h w", "$ c h w")

        # Encoder - 3 DownBlocks
        self.enc_level1 = DownBlock(n_feat0, feature_dim[0], n_enc_cab[0], kernel_size, reduction, bias, act, no_use_ca, first_act=True).rearrange("b t s c h w", "$ c h w")
        self.enc_level2 = DownBlock(feature_dim[0], feature_dim[1], n_enc_cab[1], kernel_size, reduction, bias, act, no_use_ca).rearrange("b t s c h w", "$ c h w")
        self.enc_level3 = DownBlock(feature_dim[1], feature_dim[2], n_enc_cab[2], kernel_size, reduction, bias, act, no_use_ca).rearrange("b t s c h w", "$ c h w")

        # Skip Connections - 3 SkipBlocks
        self.skip_attn1 = SkipBlock(n_feat0, n_skip_cab[0], kernel_size, reduction, bias, act, no_use_ca).rearrange("b t s c h w", "$ c h w")
        self.skip_attn2 = SkipBlock(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias, act, no_use_ca).rearrange("b t s c h w", "$ c h w")
        self.skip_attn3 = SkipBlock(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias, act, no_use_ca).rearrange("b t s c h w", "$ c h w")

        # Bottleneck
        self.bottleneck = nn.Sequential(*[CAB(feature_dim[2], kernel_size, reduction, bias, act, no_use_ca).rearrange("b t s c h w", "$ c h w")
                                          for _ in range(n_bottleneck_cab)])
        # Decoder - 3 UpBlocks
        self.prompt_level3 = PromptBlock(prompt_dim[2], len_prompt[2], prompt_size[2], feature_dim[2], learnable_prompt).rearrange("b t s c h w", "$ c h w")
        self.dec_level3 = UpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], n_dec_cab[2], kernel_size, reduction, bias, act, no_use_ca, n_history).rearrange("b t s c h w", "$ c h w")

        self.prompt_level2 = PromptBlock(prompt_dim[1], len_prompt[1], prompt_size[1], feature_dim[1], learnable_prompt).rearrange("b t s c h w", "$ c h w")
        self.dec_level2 = UpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], n_dec_cab[1], kernel_size, reduction, bias, act, no_use_ca, n_history).rearrange("b t s c h w", "$ c h w")

        self.prompt_level1 = PromptBlock(prompt_dim[0], len_prompt[0], prompt_size[0], feature_dim[0], learnable_prompt).rearrange("b t s c h w", "$ c h w")
        self.dec_level1 = UpBlock(feature_dim[0], n_feat0, prompt_dim[0], n_dec_cab[0], kernel_size, reduction, bias, act, no_use_ca, n_history).rearrange("b t s c h w", "$ c h w")

        # OutConv
        self.conv_last = conv(n_feat0, out_chans, 5, bias=bias).rearrange("b t s c h w", "$ c h w")

    def forward(self, x, history_feat: Optional[List[torch.Tensor]] = None):
        if history_feat is None:
            history_feat = [None, None, None]
        history_feat3, history_feat2, history_feat1 = history_feat
        current_feat = []

        # 0. featue extraction
        x = self.feat_extract(x)

        # 1. encoder
        x, enc1 = self.enc_level1(x)
        x, enc2 = self.enc_level2(x)
        x, enc3 = self.enc_level3(x)

        # 2. bottleneck
        x = self.bottleneck(x)

        # 3. decoder
        current_feat.append(x.clone())
        dec_prompt3 = self.prompt_level3(x)
        x = self.dec_level3(x, dec_prompt3, self.skip_attn3(enc3), history_feat3)

        current_feat.append(x.clone())
        dec_prompt2 = self.prompt_level2(x)
        x = self.dec_level2(x, dec_prompt2, self.skip_attn2(enc2), history_feat2)

        current_feat.append(x.clone())
        dec_prompt1 = self.prompt_level1(x)
        x = self.dec_level1(x, dec_prompt1, self.skip_attn1(enc1), history_feat1)

        # 4. last conv
        if self.n_history > 0:
            for i, history_feat_i in enumerate(history_feat):
                if history_feat_i is None:  # for the first cascade, repeat the current feature
                    history_feat[i] = torch.cat([torch.tile(current_feat[i], [1] * (x.ndim - 3) + [self.n_history] + [1] * 2 )], dim=-3)
                else:  # for the rest cascades: pop the oldest feature and append the current feature
                    history_feat[i] = torch.cat([current_feat[i], history_feat[i][..., :-self.feature_dim[2-i], :, :] ], dim=-3)
        return self.conv_last(x), history_feat