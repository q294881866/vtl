import math
import os
import time

import GlobalConfig
import torch
from PIL import Image
from einops import rearrange
from torch import Tensor

from dataset.DFTL import get_dataloader
from dataset.Davis2016TL import get_inpainting_dataloader
from dmac.test_dmac import get_dmac, test_dmac
from layer import helper
from layer.helper import to_image
from layer.localizator import Localizator
from util.logUtil import logger

device = torch.device("cuda:0")


def to_vrf_masks(outs):
    b, h, w = outs.shape
    for i in range(b):
        for j in range(h):
            for k in range(w):
                if outs[i, j, k] > 0.5:
                    outs[i, j, k] = 1
                else:
                    outs[i, j, k] = 0
    return outs


def test_g(data_type: int, path, test_op=-1):
    dmac = get_dmac(device, r'./model/DMAC-adv.pth')
    net_g = Localizator()
    if data_type == 0:
        dataloader = get_dataloader(set_path=os.path.join(path, GlobalConfig.TEST),
                                    batch_size=1,
                                    mode=GlobalConfig.TEST, test_op=test_op)
        net_g.load_state_dict(torch.load('./model/human/net_g.pth', map_location=device))
    else:
        dataloader = get_inpainting_dataloader(set_path=os.path.join(path, GlobalConfig.TEST),
                                               batch_size=1,
                                               mode=GlobalConfig.TEST, test_op=test_op)
        net_g.load_state_dict(torch.load('./model/inpainting/net_g.pth', map_location=device))
    net_g = net_g.to(device)
    net_g.eval()
    vrf_total = sum([param.nelement() for param in net_g.parameters()])
    dmac_total = sum([param.nelement() for param in dmac.parameters()])
    logger.info(
        "Number of parameter: vrf,%.2fM" % (vrf_total / 1e6) + "Number of parameter: dmac,%.2fM" % (dmac_total / 1e6))
    vrf_iou, dmac_iou, count, vrf_time, dmac_time = 0, 0, 0, 0, 0
    for idx, (label, _, src_images, fake_images, src, fakes, masks) in enumerate(dataloader):
        count += 1
        vrf_out, vrf_during = test_vrf(fakes, net_g, src)
        dmac_out, dmac_during = test_dmac(dmac, src_images, fake_images, device)
        masks = rearrange(masks, 'b c t ... -> (b c t) ...')
        merge_pic_test(fake_images, vrf_out, dmac_out, masks, f'test{test_op}/images/{idx}_{data_type}_mask.jpg')
        vrf_iou += mask_iou(vrf_out, masks)
        dmac_iou += mask_iou(dmac_out, masks)
        vrf_time += vrf_during
        dmac_time += dmac_during
        logger.info(
            f'idx:{idx}, vrf_iou:{round(vrf_iou / count, 3)}-{round(vrf_during, 3)}s,dmac_iou:{round(dmac_iou / count, 3)}-{round(dmac_during, 3)}s')
    logger.info(
        f'type:{data_type} vrf_iou:{round(vrf_iou / count, 3)}-{round(vrf_time/count, 3)}s,dmac_iou:{round(dmac_iou / count, 3)}-{round(dmac_time/count, 3)}s')


def test_vrf(fakes, net_g, src):
    fakes = helper.cb2b(fakes, device)
    src = helper.cb2b(src, device)
    # generator
    start = time.time()
    g = net_g([src, fakes])
    out = rearrange(g, 'b t h w -> (b t) h w').clone().detach().cpu()
    during = time.time() - start
    return to_vrf_masks(out), during


def merge_pic_test(fake_images, vrf: Tensor, dmac: Tensor, mask: Tensor, name):
    b, h, w = dmac.shape
    img = Image.open(fake_images[0][0])
    batch = 4
    images = Image.new(img.mode, (w * b, h * batch))
    for j in range(b * batch):
        if j % batch == 0:
            img = Image.open(fake_images[j // batch][0])
            img = img.resize((224, 224))
        else:
            if j % batch == 1:
                img = mask[j // batch]
            elif j % batch == 2:
                img = dmac[j // batch]
            elif j % batch == 3:
                img = vrf[j // batch]  # ours
            img = to_image(img)
        images.paste(img, box=(w * (j // batch), (j % batch) * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def mask_iou(mask1, mask2):
    area1 = mask1.sum()
    area2 = mask2.sum()
    inter = ((mask1 + mask2) == 2).sum()
    m_iou = inter / (area1 + area2 - inter)
    m_iou = round(m_iou.item(), 3)
    return 0 if math.isnan(m_iou) else mv_iou


if __name__ == '__main__':
    for j in [1]:
        path_ = '../vrf' if j == 0 else '../inpainting'
        test_g(j, path_, -1)
