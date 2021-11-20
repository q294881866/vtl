import torch
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

import GlobalConfig
from layer.block import Transformer, Residual, LeFF, UpperSample, b16
from util import figureUtil


class ConvEncoder(nn.Module):
    def __init__(self, num_frames=GlobalConfig.NUM_FRAME, in_channels=3,
                 hidden_size=14, dim=192, depth=8, heads=6, dim_head=64, dropout=0., scale_dim=4, ):
        super(ConvEncoder, self).__init__()

        self.to_conv_embedding = nn.Sequential(
            Rearrange('b t c h w -> (b t) c h w'),
            b16(dim, nn.GELU, in_channels),
            Rearrange('(b t) d h w -> b t (h w) d', t=num_frames)
        )

        # position feature
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, hidden_size ** 2, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        # train conv_block
        self.conv_block = Residual(nn.Sequential(
            LeFF(dim=dim, hw=hidden_size),
            nn.Dropout()
        ), 0.)

    def forward(self, x):
        x = self.to_conv_embedding(x)
        b, t, n, d = x.shape

        pos_embedding = repeat(self.pos_embedding, '() t n d -> b t n d', b=b)
        x += pos_embedding
        x = rearrange(x, 'b t ... -> (b t) ...')
        x = self.transformer(x)

        return self.conv_block(x)


class ConvDecoder(nn.Module):
    def __init__(self, image_size=224, num_frames=GlobalConfig.NUM_FRAME,
                 hidden_size=14, dim=192, depth=8, heads=6, dim_head=64, dropout=0., scale_dim=4, ):
        super(ConvDecoder, self).__init__()
        self.image_size = image_size
        self.num_frames = num_frames
        self.hidden_size = hidden_size

        # position feature
        self.pos_embedding = nn.Parameter(torch.randn(1, hidden_size ** 2, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        # uppers
        self.upper_samples = nn.ModuleList([])
        for i in range(4):
            self.upper_samples.append(
                UpperSample(dim // (2 ** i), dim // (2 ** (i + 1)), hw=hidden_size * (2 ** i))
            )

        # to 1 channels
        self.to_img = nn.Sequential(
            Rearrange('(b t) (h w) -> b t h w', t=self.num_frames, h=self.image_size, w=self.image_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, n, d = x.shape
        pos_embedding = repeat(self.pos_embedding, '() n d -> b n d', b=b)
        x += pos_embedding
        x = self.transformer(x)

        for upper in self.upper_samples:
            x = upper(x)

        x = x.mean(2)
        x = self.to_img(x)
        return x


class ConvTransGenerator(nn.Module):
    def __init__(self):
        super(ConvTransGenerator, self).__init__()
        self.encode = ConvEncoder()
        self.decode = ConvDecoder()

    def forward(self, x):
        src = self.encode(x[0])
        fake = self.encode(x[1])
        x = src + fake
        x = self.decode(x)
        return x
