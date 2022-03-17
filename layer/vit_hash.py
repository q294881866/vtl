import torch
import torch.nn as nn
from einops import repeat, rearrange

from config import BaseConfig
from layer.block import Transformer
from layer.helper import tensor_to_binary
from layer.swin_vit import SwinTransformer


class SpacialTemporalViT(nn.Module):
    def __init__(self, num_frames, hash_bits=1024, dim=128, depth=4, heads=3, dim_head=64, dropout=0., emb_dropout=0., scale_dim=4, ):
        super(SpacialTemporalViT, self).__init__()
        self.num_frames = num_frames
        # space feature
        self.space_transformer = SwinTransformer(embed_dim=dim, num_classes=hash_bits)
        # temporal feature
        self.temporal_token = nn.Parameter(torch.randn(1, num_frames, hash_bits))
        self.temporal_transformer = Transformer(hash_bits, depth, heads, dim_head, hash_bits * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.space_transformer(x)

        temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = rearrange(x, '(b t) ... -> b t ...', t=self.num_frames)
        x = x + temporal_tokens
        x = self.temporal_transformer(x)

        return x.mean(1)


class ViTHash(nn.Module):
    def __init__(self, num_frames=BaseConfig.NUM_FRAMES, hash_bits=BaseConfig.HASH_BITS,
            dim=BaseConfig.ALL_DIM):
        super(ViTHash, self).__init__()
        self.num_frames = num_frames
        self.feature_vit = SpacialTemporalViT(num_frames, dim=dim, hash_bits=hash_bits, )
        self.hash_head = nn.Sequential(
            nn.LayerNorm(hash_bits),
            nn.Linear(hash_bits, hash_bits),
            HashAct(act='tahn')
        )

    def forward(self, x):
        x = self.feature_vit(x)
        h = self.hash_head(x)
        return h


class HashAct(nn.Module):
    def __init__(self, act='relu'):
        super().__init__()
        if act == 'relu':
            self.act = nn.ReLU6()
            self.cal = torch.sign
        elif act == 'tahn':
            self.act = nn.Tanh()
            self.cal = torch.sign
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
