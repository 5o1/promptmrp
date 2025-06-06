from torch import nn
from typing import List, Tuple, Optional, Union

class SensitivityModel(nn.Module):

    def __init__(
        self,
        num_adj_slices: int,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        no_use_ca: bool = False,
        mask_center: bool = True,
        learnable_prompt=False,
        use_sens_adj: bool = True,
    ):

        super().__init__()
        self.mask_center = mask_center
        self.num_adj_slices = num_adj_slices
        self.use_sens_adj = use_sens_adj
        self.norm_unet = NormPromptUnet(in_chans=2*self.num_adj_slices if use_sens_adj else 2,
                                        out_chans=2*self.num_adj_slices if use_sens_adj else 2,
                                        n_feat0=n_feat0,
                                        feature_dim=feature_dim,
                                        prompt_dim=prompt_dim,
                                        len_prompt=len_prompt,
                                        prompt_size=prompt_size,
                                        n_enc_cab=n_enc_cab,
                                        n_dec_cab=n_dec_cab,
                                        n_skip_cab=n_skip_cab,
                                        n_bottleneck_cab=n_bottleneck_cab,
                                        no_use_ca=no_use_ca,
                                        learnable_prompt=learnable_prompt,
                                        )
        self.kspace_acs_extractor = KspaceACSExtractor(mask_center)

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b = x.shape[0]
        if self.use_sens_adj:
            x = rearrange(x, 'b (adj coil) h w two -> (b coil) adj h w two', adj=self.num_adj_slices)
        else:
            x = rearrange(x, 'b adj_coil h w two -> (b adj_coil) 1 h w two')
        return x, b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.use_sens_adj:
            x = rearrange(x, '(b coil) adj h w two -> b (adj coil) h w two', b=batch_size, adj=self.num_adj_slices)
        else:
            x = rearrange(x, '(b adj_coil) 1 h w two -> b adj_coil h w two', b=batch_size)

        return x

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:

        b, adj_coil, h, w, two = x.shape
        coil = adj_coil//self.num_adj_slices
        x = x.view(b, self.num_adj_slices, coil, h, w, two)
        x = x / rss_complex(x, dim=2).unsqueeze(-1).unsqueeze(2)

        return x.view(b, adj_coil, h, w, two)

    def compute_sens(self, model: nn.Module, images: torch.Tensor, compute_per_coil: bool) -> torch.Tensor:
        bc = images.shape[0]  # batch_size * n_coils
        if compute_per_coil:
            output = []
            for i in range(bc):
                output.append(model(images[i].unsqueeze(0))[0])
            output = torch.cat(output, dim=0)
        else:
            output = model(images)[0]
        return output

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[Union[int, torch.Tensor]] = None,
        mask_type: Tuple[str] = ("cartesian",),
        compute_per_coil: bool = False,
    ) -> torch.Tensor:
        masked_kspace_acs = self.kspace_acs_extractor(masked_kspace, mask, num_low_frequencies, mask_type)
        # convert to image space
        images, batches = self.chans_to_batch_dim(ifft2c(masked_kspace_acs))

        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.compute_sens(self.norm_unet, images, compute_per_coil), batches)
        )