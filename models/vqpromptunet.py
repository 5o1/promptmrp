from typing import List, Optional
import torch
from torch import nn
from .modules.conv import conv, DownBlock, CABChain, CAB, Upwohis, UpBlock
from .modules.prompt import VQPromptBlock
from .modules.transformer import MultiheadCrossAttention
from .modules import rarrange

class TimeDepthFuseBlock(nn.Module):
    def __init__(self, channels: int, n_cab:int, kernel_size:int, reduction, bias, act, no_use_ca):
        super().__init__()
        self.tatt = MultiheadCrossAttention(channels, n_head=1, n_head_feat=channels, dropout=0.0).rearrange("b t s c h w", "$ t c")
        self.datt = MultiheadCrossAttention(channels, n_head=1, n_head_feat=channels, dropout=0.0).rearrange("b t s c h w", "$ s c")
        self.decoder = CABChain(channels,n_cab,kernel_size,reduction,bias,act,no_use_ca).rearrange("b t s c h w", "$ c h w")

    def forward(self, x: torch.Tensor):
        """
        x : b t s c h w
        """
        res = self.tatt(x)
        res = self.datt(res)
        res = self.decoder(res)
        res = res + x
        return x


class VQPromptUnet(nn.Module):
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
                adaptive_input=False,
                n_buffer=0,
                n_history=0,
                 ):
        super().__init__()
        act = nn.PReLU()

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
        self.skip_attn1 = TimeDepthFuseBlock(n_feat0, n_skip_cab[0], kernel_size, reduction, bias, act, no_use_ca)
        self.skip_attn2 = TimeDepthFuseBlock(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias, act, no_use_ca)
        self.skip_attn3 = TimeDepthFuseBlock(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias, act, no_use_ca)

        # Bottleneck
        self.bottleneck = TimeDepthFuseBlock(feature_dim[2], n_bottleneck_cab, kernel_size, reduction, bias, act, no_use_ca)
        # Decoder - 3 UpBlocks
        self.prompt_level3 = VQPromptBlock(feature_dim[2], prompt_dim[2], int[feature_dim[2] * 1.3],len_prompt[2], 1, 1, reduction, prompt_size[2],0.99,kernel_size,bias, act).rearrange("b t s c h w", "$ c h w")
        self.dec_level3 = UpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], n_dec_cab[2], kernel_size, reduction, bias, act, no_use_ca, n_history).rearrange("b t s c h w", "$ c h w")

        self.prompt_level2 = VQPromptBlock(feature_dim[1], prompt_dim[1], int[feature_dim[1] * 1.3],len_prompt[2], 1, 1, reduction, prompt_size[1],0.99,kernel_size,bias, act).rearrange("b t s c h w", "$ c h w")
        self.dec_level2 = UpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], n_dec_cab[1], kernel_size, reduction, bias, act, no_use_ca, n_history).rearrange("b t s c h w", "$ c h w")

        self.prompt_level1 = VQPromptBlock(feature_dim[0], prompt_dim[0], int[feature_dim[0] * 1.3],len_prompt[2], 1, 1, reduction, prompt_size[0],0.99,kernel_size,bias, act).rearrange("b t s c h w", "$ c h w")
        self.dec_level1 = UpBlock(feature_dim[0], n_feat0, prompt_dim[0], n_dec_cab[0], kernel_size, reduction, bias, act, no_use_ca, n_history).rearrange("b t s c h w", "$ c h w")

        # OutConv
        self.conv_last = conv(n_feat0, out_chans, 5, bias=bias).rearrange("b t s c h w", "$ c h w")

    def forward(self, x: torch.Tensor, history_feat: Optional[List[torch.Tensor]] = None):
        """
        x : b t s c h w
        """
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
        dec_prompt3 = self.prompt_level3(dec_prompt3)
        x = self.dec_level3(x, dec_prompt3, self.skip_attn3(enc3), history_feat3)

        current_feat.append(x.clone())
        dec_prompt2 = self.prompt_level2(dec_prompt2)
        x = self.dec_level2(x, dec_prompt2, self.skip_attn2(enc2), history_feat2)

        current_feat.append(x.clone())
        dec_prompt1 = self.prompt_level1(dec_prompt1)
        x = self.dec_level1(x, dec_prompt1, self.skip_attn1(enc1), history_feat1)

        # 4. last conv
        if self.n_history > 0:
            for i, history_feat_i in enumerate(history_feat):
                if history_feat_i is None:  # for the first cascade, repeat the current feature
                    history_feat[i] = torch.cat([torch.tile(current_feat[i], [1] * len(x.ndim - 3) + [self.n_history] + [1] * 2 )], dim=-3)
                else:  # for the rest cascades: pop the oldest feature and append the current feature
                    history_feat[i] = torch.cat([current_feat[i], history_feat[i][..., :-self.feature_dim[2-i], :, :] ], dim=-3)
        return self.conv_last(x), history_feat