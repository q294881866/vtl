import torch
from timm.models import Xception


class XceptionFeature(Xception):
    def __init__(self, num_classes=1000, in_chans=3, drop_rate=0., global_pool='avg'):
        self.num_features = 2048
        self.out_image_size = 7
        super(XceptionFeature, self).__init__(num_classes, in_chans, drop_rate, global_pool)

    def forward(self, x):
        x = self.forward_features(x)
        return x


if __name__ == '__main__':
    img = torch.randn((4, 3, 224, 224))
    net = XceptionFeature()
    x = net(img)
    print(x.shape)
