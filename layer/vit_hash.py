import torch
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from config import BaseConfig
from layer.block import Transformer, LinearBn, Attention, Residual
from layer.helper import tensor_to_binary


class FeatureNet(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, dim=192, depth=4, heads=3,
                 in_channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., scale_dim=4, ):
        super(FeatureNet, self).__init__()

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
        self.to_param = nn.Sequential(
            Rearrange('b t d -> b d t'),
            nn.Linear(num_frames + 1, in_channels * num_frames),
            nn.GELU(),
            Rearrange('b d t -> b t d')
        )

    def forward(self, x):
        # 从[b,16,3,224,224]->[b,16,3,7,32,7,32]->[b,16,7,7,32,32,3]=[b,16,49,32x32x3]
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        return self.to_param(x)


class Discriminator(nn.Module):
    def __init__(self, dim, num_classes=2, depth=4, out_act=nn.Sigmoid):
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
                self.blocks.append(nn.Sigmoid())
        self.blocks = torch.nn.Sequential(*self.blocks)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim // 2 * 4),
            nn.Linear(dim // 2 * 4, num_classes),
            out_act()
        )

    def forward(self, x):
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x


class ViTHash(nn.Module):
    def __init__(self, image_size, patch_size, num_frames=BaseConfig.NUM_FRAMES, hash_bits=BaseConfig.HASH_BITS,
                 dim=BaseConfig.ALL_DIM, num_classes=1):
        super(ViTHash, self).__init__()
        self.feature_exact = FeatureNet(image_size, patch_size, num_frames, depth=6, heads=9)
        self.discriminate = Discriminator(dim, num_classes, out_act=nn.Softmax)
        self.hash_net = Discriminator(dim, hash_bits, out_act=nn.Softsign)

    def forward(self, x):
        x = self.feature_exact(x)
        if self.training:
            d = self.discriminate(x)
            h = self.hash_net(x)
            return d, tensor_to_binary(h)
        else:
            h = self.hash_net(x)
            return tensor_to_binary(h)
