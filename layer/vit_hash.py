import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from config import BaseConfig
from layer.block import Transformer, Residual, LinearBn, Attention
from layer.helper import tensor_to_binary


class SpacialTemporalViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, dim=192, depth=4, heads=3,
            in_channels=3, dim_head=64, dropout=0.,
            emb_dropout=0.1, scale_dim=4, ):
        super(SpacialTemporalViT, self).__init__()

        # split image to patch, embedded num_frames * patch
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        # space feature
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        self.space_token = nn.Parameter(torch.randn(1, num_patches, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        # temporal feature
        self.temporal_token = nn.Parameter(torch.randn(1, num_frames, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)

    def logits(self, x, t, n):
        h = int(np.sqrt(n))
        x = rearrange(x, 'b (h w) d -> b d h w', h=h)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = rearrange(x, '(b t) d -> b t d', t=t)
        return x

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, d = x.shape

        space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = x + space_tokens
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)

        x = self.logits(x, t, n)
        temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = x + temporal_tokens

        x = self.temporal_transformer(x)
        x = self.dropout(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, dim, num_classes=2, depth=4):
        super(Discriminator, self).__init__()
        self.blocks = []
        for i in range(1, depth):
            in_dim = dim // 2 * (1 + i)
            out_dim = dim // 2 * (2 + i)
            self.blocks.append(
                Residual(nn.Sequential(
                    LinearBn(in_dim, in_dim * 2),
                    nn.Hardswish(),
                    LinearBn(in_dim * 2, in_dim),
                ), 0.))
            if i < depth - 1:
                self.blocks.append(Attention(
                    in_dim, out_dim))
            else:
                self.blocks.append(Attention(
                    in_dim, dim))
        self.blocks = torch.nn.Sequential(*self.blocks)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            HashAct(act='tahn')
        )

    def forward(self, x):
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x


class ViTHash(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_frames=BaseConfig.NUM_FRAMES, hash_bits=BaseConfig.HASH_BITS,
            dim=BaseConfig.ALL_DIM, ):
        super(ViTHash, self).__init__()
        self.num_frames = num_frames
        self.feature_vit = SpacialTemporalViT(image_size, patch_size, num_frames, dim=dim)
        self.discriminator = Discriminator(dim, num_classes=hash_bits)

    def forward(self, x):
        x = self.feature_vit(x)
        h = self.discriminator(x)
        return h


class HashAct(nn.Module):
    def __init__(self, act='relu'):
        super().__init__()
        if act == 'relu':
            self.act = nn.ReLU6()
            self.cal = torch.sign
        elif act == 'tahn':
            self.act = nn.Tanh()
            self.cal = tensor_to_binary
        else:
            self.act = nn.Identity()
            self.cal = tensor_to_binary

    def forward(self, x):
        x = self.act(x)
        return self.cal(x)


if __name__ == '__main__':
    img = torch.randn((4, BaseConfig.NUM_FRAMES, 3, 224, 224))
    net = ViTHash()
    x = net(img)
    print(x.shape)
