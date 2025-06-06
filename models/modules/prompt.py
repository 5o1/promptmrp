import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .conv import conv, CABChain
from .vqvae import Quantize

def normalized_entropy(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized entropy of a tensor.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Normalized entropy of the input tensor.
    """
    entropy = -torch.sum(x * torch.log(x + 1e-12), dim=1, keepdim=True)
    num_classes = x.size(1)
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

class FallbackPromptBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, learnable_prompt=False):
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len - 1, prompt_dim, prompt_size, prompt_size), 
                                         requires_grad=learnable_prompt)
        self.fallback_param = nn.Parameter(torch.rand(1, prompt_dim, prompt_size, prompt_size),
                                         requires_grad=True)
        self.linear_layer = nn.Linear(lin_dim, prompt_len - 1)
        self.dec_conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        fallback_weights = normalized_entropy(prompt_weights)

        prompt_param = self.prompt_param.repeat(B, 1, 1, 1, 1)
        fallback_param = self.fallback_param.repeat(B, 1, 1, 1)

        prompt = rearrange(prompt_weights, 'B T -> B T 1 1 1') * prompt_param 
        fallback = rearrange(fallback_weights, 'B C -> B C 1 1')
        fallback = fallback * fallback_param
        prompt = torch.sum(prompt, dim=1)
        prompt = prompt + fallback

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.dec_conv3x3(prompt)

        return prompt
    
    
class VQPromptBlock(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        hidden_channels=64,
        embed_channels=64,
        n_enc_cab = 3,
        n_dec_cab = 3,
        reduction=4,
        n_embed=512,
        decay=0.99,
        kernel_size = 3,
        bias = False,
        act = nn.PReLU(),
    ):
        super().__init__()

        self.enc = nn.Sequential(
            conv(in_channels, hidden_channels, 1),
            CABChain(hidden_channels, n_enc_cab, kernel_size, reduction, bias=bias, act=act),
            conv(hidden_channels, embed_channels, 1),
        )
        self.quantize = Quantize(embed_channels, n_embed, decay=decay)
        self.dec = nn.Sequential(
            conv(embed_channels, hidden_channels, 1),
            CABChain(hidden_channels, n_dec_cab, kernel_size, reduction, bias=bias, act=act),
            conv(hidden_channels, out_channels, 1, bias=bias)
        )

    def forward(self, input: torch.Tensor):
        quant = self.enc(input).permute(0, 2, 3, 1) # B H W C
        quant, diff, id = self.quantize(quant)
        quant = quant.permute(0, 3, 1, 2) # B C H W
        quant = self.dec(quant)
        diff = diff.unsqueeze(0)
        return quant, diff

    def decode_code(self, code):
        quant = self.quantize.embed_code(code)
        quant = quant.permute(0, 3, 1, 2)

        dec = self.dec(quant)
        return dec