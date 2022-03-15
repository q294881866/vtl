import _thread
import argparse
import os

import torch.nn as nn

from config import DFTLConfig, Davis2016Config, FFConfig
from dataset.Base import TrainItem, get_dataloader, TrainCache, load_cache
from dataset.DFTL import DFTLDataset
from dataset.Davis2016TL import Davis2016Dataset
from dataset.FF import FFDataset
from layer import helper, fn
from layer.genesis import Genesis
from layer.helper import cb2b
from util import figureUtil
from util.logUtil import logger

# program init
label_set = {}
bce_loss = nn.BCELoss(reduction='mean')
choices = {
    'DFTL': (DFTLDataset, DFTLConfig),
    'Davis2016': (Davis2016Dataset, Davis2016Config),
    'FF': (FFDataset, FFConfig),
}


def train(cfg, dataloader_, test_loader_):
    genesis = Genesis(cfg, train_h=True)

    test_itr = enumerate(test_loader_)
    for epoch in range(cfg.EPOCH):
        train_cache = TrainCache(size=32)
        _thread.start_new_thread(load_cache, (dataloader_, train_cache))
        while not train_cache.finished:
            if train_cache.has_item():
                idx, item = train_cache.next_data()
                train_step(genesis, item, idx, epoch)
                test_step(genesis, idx, epoch, test_itr)


def train_step(genesis: Genesis, item: TrainItem, idx, epoch):
    # HashNet
    src = cb2b(item.p1, genesis.device)
    fake = cb2b(item.p2, genesis.device)
    mask = cb2b(item.p3, genesis.device)
    loss_g, g = train_g(genesis, [src, fake], mask, idx)
    # epoch log
    logger.info("Train Epoch:{}/{},G Loss:{:.5f}".format(epoch, idx, loss_g))


def test_step(genesis: Genesis, idx, epoch, test_itr, device):
    if idx % 100 == 0:
        genesis.eval()
        _, (sources, fakes, masks) = test_itr.__next__()
        # HashNet
        fakes = cb2b(fakes, device)
        sources = cb2b(sources, device)
        masks = cb2b(masks, device)
        g = genesis.g([sources, fakes])
        # save generate p3
        figureUtil.merge_pic(g, masks, 'images/{}_{}_{}_test.jpg'.format(epoch, idx, 0))
        # save generate p3
        genesis.save('models/{}_{}_'.format(epoch, idx))
        genesis.train()


def train_g(genesis: Genesis, train_data, masks, idx):
    # train
    try:
        g = genesis.g(train_data)
        g_loss = fn.mask_loss(g, masks)
        # backward
        genesis.reset_grad()
        g_loss.backward()
        genesis.opt_g.step()
        if idx % 100 == 0:
            figureUtil.merge_pic(g, masks, 'images/{}_{}_mask.jpg'.format(idx, 0))
        return g_loss, g
    except Exception as e:
        print(e)


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r'E:\dataset\ff')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--type', type=str, default='FF')
parser.add_argument('--bits', type=int, default=512)
parser.add_argument('--pretrained', type=str, default='net_h.pth')
if __name__ == '__main__':
    args_ = parser.parse_args()
    print('args:{}'.format(args_))
    helper.set_hash_bits(args_.bits)
    Dataset, cfg = choices[args_.type]

    train_cfg = cfg(cfg.TRAIN, os.path.join(args_.path, cfg.TRAIN), args_.path, args_.local_rank)
    dataset = Dataset(cfg=train_cfg)
    dataloader = get_dataloader(dataset=dataset, cfg=train_cfg)
    train_cfg.pretrained = args_.pretrained

    test_cfg = cfg(cfg.TEST, os.path.join(args_.path, cfg.TEST), args_.path, args_.local_rank)
    dataset = Dataset(cfg=test_cfg)
    test_loader = get_dataloader(dataset=dataset, cfg=test_cfg)


    train(train_cfg, dataloader, test_loader)