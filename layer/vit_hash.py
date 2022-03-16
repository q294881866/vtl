import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from config import BaseConfig
from layer.block import Transformer
from layer.helper import tensor_to_binary
from layer.xception import XceptionFeature


class SpacialTemporalViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, dim=192, depth=4, heads=3,
            in_channels=3, dim_head=64, dropout=0.,
            emb_dropout=0., scale_dim=4, ):
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

        x = x + self.pos_embedding

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)

        x = self.logits(x, t, n)
        temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = x + temporal_tokens

        x = self.temporal_transformer(x)

        return x.mean(1)


class ViTHash(nn.Module):
    def __init__(self, image_size, patch_size, num_frames=BaseConfig.NUM_FRAMES, hash_bits=BaseConfig.HASH_BITS,
            dim=BaseConfig.ALL_DIM, cnn=True):
        super(ViTHash, self).__init__()
        self.num_frames = num_frames
        self.feature_exact = XceptionFeature()
        if cnn:
            dim = self.feature_exact.num_features
            image_size = self.feature_exact.out_image_size
            patch_size = 1
        self.feature_vit = SpacialTemporalViT(image_size, patch_size, num_frames, dim=dim, in_channels=dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hash_bits),
            HashAct(act='tahn')
        )

    def forward(self, x):
        x = rearrange(x, 'b t ... -> (b t) ...')
        x = self.feature_exact(x)
        x = rearrange(x, '(b t) ... -> b t ...', t=self.num_frames)
        x = self.feature_vit(x)
        h = self.mlp_head(x)
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
    net = ViTHash(7, 1)
    x = net(img)
    print(x.shape)
