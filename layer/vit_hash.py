import torch
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from config import BaseConfig
from layer.block import Transformer, LinearBn, Attention, Residual
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
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        # temporal feature
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        return x.mean(1)


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
        self.blocks = torch.nn.Sequential(*self.blocks)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim // 2 * 4),
            nn.Linear(dim // 2 * 4, num_classes),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x


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
            HashAct()
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
        else:
            self.act = nn.Tanh()
            self.cal = tensor_to_binary

    def forward(self, x):
        x = self.act(x)
        return self.cal(x)


if __name__ == '__main__':
    img = torch.randn((4, BaseConfig.NUM_FRAMES, 3, 224, 224))
    net = ViTHash(7, 1)
    x = net(img)
    print(x.shape)
