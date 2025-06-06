from torch import nn
import torch
from typing import Optional

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, stride=stride)

class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, n_cab, kernel_size, reduction, bias, act,
                 no_use_ca=False, n_history=0):
        super().__init__()
        # momentum layer
        self.n_history = n_history
        if n_history > 0:
            self.momentum = nn.Sequential(
                nn.Conv2d(in_dim*(n_history+1), in_dim, kernel_size=1, bias=bias),
                CAB(in_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
            )

        self.fuse = nn.Sequential(*[CAB(in_dim+prompt_dim, kernel_size, reduction,
                                        bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        self.reduce = nn.Conv2d(in_dim+prompt_dim, in_dim, kernel_size=1, bias=bias)

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False))

        self.ca = CAB(out_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

    def forward(self, x, prompt_dec, skip, history_feat: Optional[torch.Tensor] = None):
        # momentum layer
        if self.n_history > 0:
            if history_feat is None:
                x = torch.cat([torch.tile(x, (1, self.n_history+1, 1, 1))], dim=1)
            else:
                x = torch.cat([x, history_feat], dim=1)

            x = self.momentum(x)

        x = torch.cat([x, prompt_dec], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)

        x = self.up(x) + skip
        x = self.ca(x)

        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, no_use_ca=False):
        super().__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        if not no_use_ca:
            self.CA = CALayer(n_feat, reduction, bias=bias)
        else:
            self.CA = nn.Identity()
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class Upwohis(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, n_cab, kernel_size, reduction, bias, act):
        super().__init__()
        self.fuse = nn.Sequential(*[CAB(in_dim+prompt_dim, kernel_size, reduction,
                                        bias=bias, act=act) for _ in range(n_cab)])
        self.reduce = nn.Conv2d(in_dim+prompt_dim, in_dim, kernel_size=1, bias=bias)

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False))

        self.ca = CAB(out_dim, kernel_size, reduction, bias=bias, act=act)

    def forward(self, x, prompt_dec, skip):
        x = torch.cat([x, prompt_dec], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)

        x = self.up(x) + skip
        x = self.ca(x)
        return x
    

class JustUp(nn.Module):
    def __init__(self, in_dim, out_dim, n_cab, kernel_size, reduction, bias, act):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False))

        self.ca = CABChain(out_dim, n_cab, kernel_size, reduction, bias, act)
        
    def forward(self, x, skip):
        x = self.up(x) + skip
        x = self.ca(x)
        return x
    

class CABChain(nn.Module):
    def __init__(self, enc_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False, is_res = False):
        super().__init__()
        self.n_cab = n_cab
        self.is_res = is_res

        if n_cab == 0:
            self.skip_attn = nn.Identity()
        else:
            self.skip_attn = nn.Sequential(*[CAB(enc_dim, kernel_size, reduction, bias=bias, act=act,
                                                 no_use_ca=no_use_ca) for _ in range(n_cab)])

    def forward(self, x):
        res = self.skip_attn(x)
        if self.n_cab > 1 and self.is_res:
            res += x
        return res
    
class DownBlock(nn.Module):
    def __init__(self, input_channel, output_channel, n_cab, kernel_size, reduction, bias, act,
                 no_use_ca=False, first_act=False):
        super().__init__()
        if first_act:
            self.encoder = [CAB(input_channel, kernel_size, reduction,bias=bias, act=nn.PReLU(), no_use_ca=no_use_ca)]
            self.encoder = nn.Sequential(
                    *(self.encoder+[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) 
                                    for _ in range(n_cab-1)]))
        else:
            self.encoder = nn.Sequential(
                *[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) 
                  for _ in range(n_cab)])
        self.down = nn.Conv2d(input_channel, output_channel,kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        enc = self.encoder(x)
        x = self.down(enc)
        return x, enc
