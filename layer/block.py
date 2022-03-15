import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class LeFF(nn.Module):

    def __init__(self, dim=192, scale=4, depth_kernel=3, hw=14, padding=1):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(scale_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=hw, w=hw)
                                     )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=padding, groups=scale_dim, bias=False),
            nn.MaxPool2d(kernel_size=depth_kernel, stride=1, padding=padding),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=hw, w=hw)
        )

        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                       Rearrange('b n c -> b c n'),
                                       nn.BatchNorm1d(dim),
                                       nn.GELU(),
                                       Rearrange('b c n -> b n c')
                                       )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x


class UpperSample(nn.Module):

    def __init__(self, in_dim, out_dim, hw):
        super(UpperSample, self).__init__()

        self.up_proj = nn.Sequential(nn.Linear(in_dim, in_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(in_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=hw, w=hw)
                                     )

        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', c=out_dim)
        )

        self.to_out = nn.Sequential(nn.Linear(out_dim, out_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(out_dim),
                                    nn.GELU(),
                                    Rearrange('b c n -> b n c'),
                                    nn.Dropout()
                                    )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.de_conv(x)
        x = self.to_out(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ReAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class LinearBn(nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.fc = nn.Linear(a, b, bias=False)
        self.bn = nn.BatchNorm1d(b)
        nn.init.constant_(self.bn.weight, bn_weight_init)

    def forward(self, x):
        x = self.fc(x)
        return self.bn(x.flatten(0, 1)).reshape_as(x)


class Attention(nn.Module):
    def __init__(self, dim, out_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
            LinearBn(dim, out_dim)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


def conv_blocks(n, activation=nn.GELU, in_channels=3):
    return torch.nn.Sequential(
        ResnetBlock(in_channels, n // 8),
        activation(),
        ResnetBlock(n // 8, n // 4),
        activation(),
        ResnetBlock(n // 4, n // 2),
        activation(),
        ResnetBlock(n // 2, n))


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
            groups=1, bn_weight_init=1):
        super().__init__()
        self.c = torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(b, momentum=0.1)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.c(x)
        x = self.bn(x)
        return x


class ResnetBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = Conv2d_BN(inplanes, inplanes * self.expansion)
        self.conv2 = Conv2d_BN(inplanes * self.expansion, inplanes * self.expansion, 3, 1, 1)
        self.conv3 = Conv2d_BN(inplanes * self.expansion, inplanes)
        self.stride = stride
        self.up = Conv2d_BN(inplanes, planes, 3, 2, 1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual

        return self.up(out)




