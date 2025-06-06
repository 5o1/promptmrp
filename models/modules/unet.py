import torch
from torch import nn
from .conv import conv, DownBlock, CABChain, JustUp
from typing import List
from . import rearrange

class Unet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels:int,
            hidden_channels:list,
            n_enc_cab: List[int],
            n_dec_cab: List[int],
            n_skip_cab: List[int],
            n_cab_bottle:int,
            kernel_size = 3,
            reduction = 4,
            bias = False
            ):
        super().__init__()
        self.depth = len(hidden_channels) - 1


        act = nn.PReLU()
        self.to_in = conv(in_channels, hidden_channels[0], kernel_size, bias=bias).rearrange("b t s c h w", "$ c h w")

        self.encoder = nn.ModuleList([
            DownBlock(hidden_channels[i], hidden_channels[i+1], n_enc_cab[i], kernel_size, reduction, bias, act, no_use_ca=False, first_act=(i==0)).rearrange("b t s c h w", "$ c h w")
            for i in range(len(hidden_channels)-1)
            ])

        self.skipper = nn.ModuleList([
            CABChain(hidden_channels[i], n_skip_cab[i], kernel_size, reduction, bias, act, no_use_ca=False, is_res=True).rearrange("b t s c h w", "$ c h w")
            for i in range(len(hidden_channels)-1)
            ])
        
        self.bottleneck = CABChain(hidden_channels[-1], n_cab_bottle, kernel_size, reduction, bias, act, no_use_ca=False).rearrange("b t s c h w", "$ c h w")
        
        self.decoder = nn.ModuleList([
            JustUp(hidden_channels[i], hidden_channels[i+1], n_dec_cab[i], kernel_size, reduction, bias, act).rearrange("b t s c h w", "$ c h w")
            for i in range(len(hidden_channels)-1)
        ])

        self.to_out = conv(hidden_channels[0], out_channels, kernel_size, bias=bias).rearrange("b t s c h w", "$ c h w")
    
    def forward(self, input: torch.Tensor):
        input = self.to_in(input)

        def level_fn(input, level):
            if level == self.depth:
                input = self.bottleneck(input)
                return input
            input, res = self.encoder[level](input)
            res = self.skipper[level](res)
            input = level_fn(input, level+1)
            input = self.decoder[level](input, res)
            return input
        
        input = self.to_out(input)
        return input