import torch
from torch import nn
from typing import Union

from ulip_manage.Model.Layer.residual_attention_block import ResidualAttentionBlock


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Union[torch.Tensor, None] = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
