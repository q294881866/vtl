import numpy as np
import torch

import GlobalConfig
import util.figureUtil
from dataset.dataset import get_dataloader
from layer import helper
from layer.ConvTransGenerator import ConvTransGenerator
from layer.genesis import HashNet
from layer.model import Generator, TransGenerator
from util import figureUtil


def test_generator():
    f = torch.randn([2, GlobalConfig.NUM_FRAME * 3, 192])
    model = Generator()
    out = model(f)
    util.figureUtil.merge_pic(out, out, './test.jpg')
    print(out.shape)


def test_trans_generator():
    src = torch.randn([GlobalConfig.BATCH_SIZE, GlobalConfig.NUM_FRAME, 3, 224, 224])
    fake = torch.randn([GlobalConfig.BATCH_SIZE, GlobalConfig.NUM_FRAME, 3, 224, 224])

    model = ConvTransGenerator()
    out = model([src, fake])
    figureUtil.merge_pic(out, out, 'test.jpg')
    print(out.shape)


def test_vrf():
    img = torch.randn([GlobalConfig.BATCH_SIZE, GlobalConfig.NUM_FRAME, 3, 224, 224]).cuda()
    label = np.random.randint(0, 2, GlobalConfig.BATCH_SIZE)
    model = HashNet(224, 16, num_classes=8).cuda()
    d, g, h = model(img)
    util.figureUtil.merge_pic(g, g, 'test.jpg')
    act_rate = helper.find_index(h, label)
    print(act_rate)


def test_dataloader():
    dataloader = get_dataloader(set_path=r'Y:\vrf_\test', batch_size=2, mode=GlobalConfig.TEST)
    test_itr = enumerate(dataloader)
    idx, (label, src_file, fake, src, mask_) = test_itr.__next__()
    print(src.shape)


if __name__ == '__main__':
    test_trans_generator()
    test_dataloader()
    # test_discriminator()
    # test_vrf()
    # test_dataloader()
    # test_train()
