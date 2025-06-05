import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from mri_utils import ifft2c, complex_mul, rss, complex_abs, rss_complex, sens_expand, sens_reduce
from einops import rearrange

from .promptmr_v2 import SensitivityModel

from .modules.conv import conv, DownBlock, CABChain, CAB, Upwohis
from .modules.prompt import VQPromptBlock

# class VQPromptUnet(nn.Module):
#     def __init__(self,
#                 in_channels: int,
#                 out_channels: int,
#                 n_feat0: int,
#                 feature_dim: List[int],
#                 prompt_dim: List[int],
#                 len_prompt: List[int],
#                 prompt_size: List[int],
#                 n_enc_cab: List[int],
#                 n_dec_cab: List[int],
#                 n_skip_cab: List[int],
#                 n_bottleneck_cab: int,
#                 kernel_size=3,
#                 reduction=4,
#                 act=nn.PReLU(),
#                 bias=False,
#                 *args, **kwargs
#                  ):
#         super().__init__()
#         self.feature_dim = feature_dim

#         # Feature extraction
#         self.feat_extract = conv(in_channels, n_feat0, kernel_size, bias=bias)

#         # Encoder - 3 DownBlocks
#         self.enc_level1 = DownBlock(n_feat0, feature_dim[0], n_enc_cab[0], kernel_size, reduction, bias, act, first_act=True)
#         self.enc_level2 = DownBlock(feature_dim[0], feature_dim[1], n_enc_cab[1], kernel_size, reduction, bias, act)
#         self.enc_level3 = DownBlock(feature_dim[1], feature_dim[2], n_enc_cab[2], kernel_size, reduction, bias, act)

#         # Skip Connections - 3 SkipBlocks
#         self.skip_attn1 = ResCABChain(n_feat0, n_skip_cab[0], kernel_size, reduction, bias, act)
#         self.skip_attn2 = ResCABChain(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias, act)
#         self.skip_attn3 = ResCABChain(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias, act)

#         # Bottleneck
#         self.bottleneck = nn.Sequential(*[CAB(feature_dim[2], kernel_size, reduction, bias, act)
#                                           for _ in range(n_bottleneck_cab)])
#         # Decoder - 3 UpBlocks
#         self.prompt_level3 = VQPromptBlock(prompt_dim[2], len_prompt[2], prompt_size[2], feature_dim[2])
#         self.dec_level3 = Upwohis(feature_dim[2], feature_dim[1], prompt_dim[2], n_dec_cab[2], kernel_size, reduction, bias, act)

#         self.vq_up3to2 = nn.ConvTranspose2d(feature_dim[2], feature_dim[1], kernel_size=2, stride=2, padding=0, bias=bias)

#         self.prompt_level2 = VQPromptBlock(prompt_dim[1], len_prompt[1], prompt_size[1], feature_dim[1])
#         self.dec_level2 = Upwohis(feature_dim[1], feature_dim[0], prompt_dim[1], n_dec_cab[1], kernel_size, reduction, bias, act)

#         self.vq_up2to1 = nn.ConvTranspose2d(feature_dim[1], feature_dim[0], kernel_size=2, stride=2, padding=0, bias=bias)

#         self.prompt_level1 = VQPromptBlock(prompt_dim[0], len_prompt[0], prompt_size[0], feature_dim[0])
#         self.dec_level1 = Upwohis(feature_dim[0], n_feat0, prompt_dim[0], n_dec_cab[0], kernel_size, reduction, bias, act)

#         # OutConv
#         self.out = conv(n_feat0, out_channels, 5, bias=bias)

#     def forward(self, x , *args, **kwargs):
#         # 0. featue extraction
#         x = self.feat_extract(x)

#         # 1. encoder
#         x, enc1 = self.enc_level1(x)
#         x, enc2 = self.enc_level2(x)
#         x, enc3 = self.enc_level3(x)

#         # 2. bottleneck
#         x = self.bottleneck(x)

#         # 3. decoder
#         dec_prompt3 = x
#         dec_prompt3 = self.prompt_level3(dec_prompt3)
#         x = self.dec_level3(x, dec_prompt3, self.skip_attn3(enc3))

#         dec_prompt2 = torch.cat([x, dec_prompt3], dim=1)
#         dec_prompt2 = self.prompt_level2(dec_prompt2)
#         x = self.dec_level2(x, dec_prompt2, self.skip_attn2(enc2))

#         dec_prompt1 = torch.cat([x, dec_prompt3], dim=1)
#         dec_prompt1 = self.prompt_level1(dec_prompt1)
#         x = self.dec_level1(x, dec_prompt1, self.skip_attn1(enc1))

#         # 4. out
#         x = self.out(x)
#         return x

class NormUnet(nn.Module):
    def __init__(
        self,
        model: nn.Module
    ):
        super().__init__()
        self.model = model
        if hasattr(model, "n_buffer"):
            self.n_buffer = model.n_buffer

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == 2
        return rearrange(x, '... c h w two -> ... (two c) h w')

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-3) % 2 == 0
        return rearrange(x, '... (two c) h w -> ... c h w two', two=2).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dims = tuple(range(1, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True)
        x = (x - mean) / std, mean, std
        return x

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor,
              h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, raw_shape: torch.Size,
                history_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                buffer: torch.Tensor = None):
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        cc = x.shape[-4]
        if buffer is not None:
            x = torch.cat([x, buffer], dim=-4)

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x, history_feat = self.model(x, history_feat = history_feat, raw_shape = raw_shape)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        if buffer is not None:
            x, _, latent, _ = torch.split(x, [cc, cc, cc, x.shape[-4] - 3*cc], dim=-4)
        else:
            latent = None
        return x, latent, history_feat

class WrappedSenseBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = NormUnet(model)
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, current_img: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        b, t, s, c, h, w, two = current_img.shape
        current_img = rearrange(current_img, "b t s c h w two -> (b t s) c h w two")
        sens_maps = rearrange(sens_maps, "b t s c h w two -> (b t s) c h w two")
        current_kspace = sens_expand(current_img, sens_maps, num_adj_slices=1)
        current_kspace = rearrange(current_kspace, "(b t s) c h w two -> b t s c h w two", b=b, t=t, s=s)
        return current_kspace

    def sens_reduce(self, masked_kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        b, t, s, c, h, w, two = masked_kspace.shape
        masked_kspace = rearrange(masked_kspace, "b t s c h w two -> (b t s) c h w two")
        sens_maps = rearrange(sens_maps, "b t s c h w two -> (b t s) c h w two")
        image = sens_reduce(masked_kspace, sens_maps, num_adj_slices=1) # (b t s) c h w two
        image = rearrange(image, "(b t s) c h w two -> b t s c h w two", b=b, t=t, s=s)
        return image

    def forward(
        self,
        current_img: torch.Tensor,
        img_zf: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        history_feat: Optional[Tuple[torch.Tensor, ...]] = None
    ):
        """
        b t s c h w two
        """
        current_kspace = self.sens_expand(current_img, sens_maps)  # b t s c h w two
        ffx = current_kspace * mask.float() + 0.0
        ffx = self.sens_reduce(ffx, sens_maps)
        if self.model.n_buffer > 0: # b t s c h w two
            # adaptive input. buffer: A^H*A*x_i, s_i, x0, A^H*A*x_i-x0
            buffer = torch.cat([ffx, latent, img_zf] + [ffx-img_zf]*(self.model.n_buffer-3), dim=-4)
        else:
            buffer = None
            
        soft_dc = (ffx - img_zf) * self.dc_weight

        b, t, s, c, h, w, _ = raw_shape = current_img.shape
        
        model_term, latent, history_feat = self.model(current_img, raw_shape, history_feat, buffer)
        img_pred = current_img - soft_dc - model_term
        return img_pred, latent, history_feat


class WrappedSensitivityModel(nn.Module):
    def __init__(
        self,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
    ):
        super().__init__()
        self.model = SensitivityModel( # wo adj to channel
            num_adj_slices=1,
            n_feat0=n_feat0,
            feature_dim=feature_dim,
            prompt_dim=prompt_dim,
            len_prompt=len_prompt,
            prompt_size=prompt_size,
            n_enc_cab=n_enc_cab,
            n_dec_cab=n_dec_cab,
            n_skip_cab=n_skip_cab,
            n_bottleneck_cab=n_bottleneck_cab,
            no_use_ca=False,
            mask_center=True,
            learnable_prompt=False,
            use_sens_adj=False,
        )
    
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        b, t, s, c, h, w, two = masked_kspace.shape
        masked_kspace = rearrange(masked_kspace, "b t s c h w two-> (b t s) c h w two")
        mask = rearrange(mask, "b t s c h w two -> (b t s) c h w two")
        sens_map = self.model(masked_kspace, mask, *args, **kwargs) # (b t s) c h w two
        sens_map = rearrange(sens_map, "(b t s) c h w two -> b t s c h w two", b=b, t=t, s=s)
        return sens_map

class VVE(nn.Module):

    def __init__(
            self,
            sens_net: nn.Module,
            cascades: List[nn.Module]
    ):
        super().__init__()
        self.sens_net = sens_net
        self.cascades = nn.ModuleList([
            WrappedSenseBlock(cascade) for cascade in cascades
        ])

    def sens_reduce(self, masked_kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        b, t, s, c, h, w, _ = masked_kspace.shape
        masked_kspace = rearrange(masked_kspace, "b t s c h w two -> (b t s) c h w two")
        sens_maps = rearrange(sens_maps, "b t s c h w two -> (b t s) c h w two")
        image = sens_reduce(masked_kspace, sens_maps, num_adj_slices=1) # (b t s) c h w two
        image = rearrange(image, "(b t s) c h w two -> b t s c h w two", b=b, t=t, s=s)
        return image

    def rss(self, img_pred: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        img_pred = complex_mul(img_pred, sens_maps)
        img_pred = complex_abs(img_pred)
        img_pred = rss(img_pred, dim=1)
        return img_pred

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: torch.Tensor,
        mask_type: Tuple[str] = ("cartesian",),
        compute_sens_per_coil: bool = False, # can further reduce the memory usage
    ) -> torch.Tensor:
        '''
        Args:
            masked_kspace: (b, t, s, c, h, w, two) complex input k-space data
            mask: (b, t, s, 1, h, w, two) or (b, 1, 1, 1, h, w, two) mask
            num_low_frequencies: (b) number of low frequencies
            mask_type: (str) mask type
            compute_sens_per_coil: (bool) whether to compute sensitivity maps per coil for memory saving
        '''
        assert masked_kspace.dim() == 7, "VVE input masked_kspace must be 6D tensor (b, t, s, c, h, w, two)."

        if masked_kspace.size(2) != mask.size(2):
            mask = mask.expand(-1, -1, masked_kspace.size(2), -1, -1, -1, -1)  # (b, t, s, 1, h, w, two)

        b, t, s, c, h, w, _ = masked_kspace.shape

        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies, mask_type, compute_sens_per_coil) # b t s c h w two

        img_zf = self.sens_reduce(masked_kspace, sens_maps) # b t s c h w two
        img_pred = img_zf.clone()
        latent = img_zf.clone()
        history_feat = None

        for cascade in self.cascades: # b t s c h w two
            img_pred, latent, history_feat = cascade(img_pred, img_zf, latent, mask, sens_maps, history_feat)

        # get central slice of rss as final output
        img_pred = img_pred[:, img_pred.size(1) // 2, img_pred.size(2) //2, :, :, :, :]  # (b, t, c, h, w, two)
        sens_maps = sens_maps[:, sens_maps.size(1) // 2, sens_maps.size(2) // 2, :, :, :, :] # (b, t, c, h, w, two)
        img_pred = self.rss(img_pred, sens_maps)  # (b, h, w)
            
        # prepare for additional output
        img_zf = masked_kspace[:, masked_kspace.size(1) // 2, masked_kspace.size(2) // 2, :, :, :]  # (b, t, c, h, w, two)
        img_zf = self.rss(img_zf, sens_maps)  # (b, h, w)
        sens_maps = torch.view_as_complex(sens_maps)  # (b, c, h, w)
        return {
            'img_pred': img_pred,
            'img_zf': img_zf,
            'sens_maps': sens_maps
        }

