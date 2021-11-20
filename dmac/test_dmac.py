import time
from abc import ABC

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .dmac_vgg_skip import DMAC_VGG


class SoftmaxMask(nn.Module, ABC):
    def __init__(self):
        super(SoftmaxMask, self).__init__()
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.softmax(x)
        return x[:, 1, :, :]


# pre-process image function
def get_input_img(img, input_scale):
    input_img1 = np.zeros((input_scale, input_scale, 3))
    img = img.astype(float)
    img = cv2.resize(img, (input_scale, input_scale)).astype(float)
    img[:, :, 0] = img[:, :, 0] - 104.008
    img[:, :, 1] = img[:, :, 1] - 116.669
    img[:, :, 2] = img[:, :, 2] - 122.675
    input_img1[:img.shape[0], :img.shape[1], :] = img

    return input_img1


def read_images(files, device):
    list_ = []
    for file in files:
        input_image = cv2.imread(file[0])
        input_image = get_input_img(input_image, 256)
        image = Variable(torch.from_numpy(input_image[np.newaxis, :].transpose(0, 3, 1, 2)).float())
        list_.append(image)
    return torch.cat(list_, dim=0).to(device)


def to_masks(outs):
    b, h, w = outs.shape
    for i in range(b):
        for j in range(h):
            for k in range(w):
                if outs[i, j, k] < 0.5:
                    outs[i, j, k] = 1
                else:
                    outs[i, j, k] = 0
    return outs


def get_dmac(device, path):
    net_g = DMAC_VGG(2, 0, 256)
    net_g = net_g.to(device)
    net_g.eval()
    net_g.load_state_dict(torch.load(path, map_location=device))
    return net_g


softmax_mask = SoftmaxMask()
upsample_layer = nn.UpsamplingBilinear2d(size=(224, 224))


def test_dmac(net_g, src_files, fake_files, device):
    src_image = read_images(src_files, device)
    fakes = read_images(fake_files, device)
    # generator
    start = time.time()
    out = net_g(src_image, fakes)
    out = upsample_layer(out[1])
    out = softmax_mask(out).clone().detach().cpu()
    out = to_masks(out)
    during = time.time() - start
    return out, during
