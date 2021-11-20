import torch
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

import GlobalConfig
import layer.helper
from layer.block import Transformer, LinearBlock, LinearBn, Attention, Residual, UpperSample


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


class Generator(nn.Module):
    def __init__(self, dim=GlobalConfig.ALL_DIM, hidden_size=14, num_frames=GlobalConfig.NUM_FRAME,
                 depth=4, in_channels=3):
        super(Generator, self).__init__()
        self.num_frames = num_frames
        self.dim = dim
        self.hidden_size = hidden_size
        self.image_size = hidden_size * (2 ** depth)
        self.l1 = nn.Sequential(
            LinearBlock(hidden_size ** 2, hidden_size ** 2),
            Rearrange('b t d n -> b n d t'),
            LinearBlock(num_frames * in_channels, num_frames),
            Rearrange('b n d t -> (b t) n d'),
        )

        # to image
        self.upper_samples = nn.ModuleList([])
        for i in range(depth):
            self.upper_samples.append(
                UpperSample(dim // (2 ** i), dim // (2 ** (i + 1)), hw=hidden_size * (2 ** i))
            )
        # to 1 channels
        self.l2 = nn.Sequential(
            Rearrange('(b t) (h w) -> b t h w', t=self.num_frames, h=self.image_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = repeat(x, 'b t d -> b t d n', n=self.hidden_size ** 2)
        x = self.l1(x)

        for upper in self.upper_samples:
            x = upper(x)

        x = x.mean(-1)
        x = self.l2(x)
        return x


class HashNet(nn.Module):
    def __init__(self, image_size, patch_size, num_frames=GlobalConfig.NUM_FRAME, hash_bits=GlobalConfig.HASH_BIT,
                 dim=GlobalConfig.ALL_DIM, num_classes=1):
        super(HashNet, self).__init__()
        self.feature_exact = FeatureNet(image_size, patch_size, num_frames, depth=6, heads=9)
        self.discriminate = Discriminator(dim, num_classes, out_act=nn.Softmax)
        self.hash_net = Discriminator(dim, hash_bits, out_act=nn.Tanh)

    def forward(self, x):
        x = self.feature_exact(x)
        if self.training:
            d = self.discriminate(x)
            h = self.hash_net(x)
            return d, h
        else:
            h = self.hash_net(x)
            return h


class TransGenerator(nn.Module):
    def __init__(self, image_size=224, patch_size=GlobalConfig.PATCH_SIZE, num_frames=GlobalConfig.NUM_FRAME):
        super(TransGenerator, self).__init__()
        self.feature_exact = FeatureNet(image_size, patch_size, num_frames, depth=8, in_channels=3)
        self.generate = Generator(in_channels=3)

    def forward(self, x):
        src = self.feature_exact(x[0])
        fake = self.feature_exact(x[1])
        x = src + fake
        x = self.generate(x)
        return x
