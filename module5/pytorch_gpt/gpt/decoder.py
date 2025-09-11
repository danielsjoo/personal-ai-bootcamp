import torch
from torch import nn
import math
    
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dense_dim: int,dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim= embed_dim, 
            num_heads = num_heads,
            batch_first = True,
            )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim,dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim,embed_dim),
        )
    
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        res = self.layer_norm1(x)
        res, _ = self.mha(
            query = res,
            key = res,
            value = res,
            key_padding_mask=key_padding_mask,
            attn_mask = causal_mask,
        )
        x = x + res
        x = self.layer_norm2(x)
        res = self.feed_forward(x)
        x = x + res
        return x