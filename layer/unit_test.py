import torch
from timm.models import SwinTransformer

from config import BaseConfig


class VisionTransformer(SwinTransformer):
    def __init__(self):
        super(VisionTransformer, self).__init__()

    def forward(self, x):
        x = self.forward_features(x)
        return x


if __name__ == '__main__':
    img = torch.randn((BaseConfig.NUM_FRAMES, 3, 224, 224))
    net = PVTransformerV2()
    x = net(img)
    print(x.shape)
