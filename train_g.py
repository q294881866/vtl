import _thread
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import GlobalConfig
from dataset.DFTL import TrainCache, load_cache, get_dataloader, TrainItem
from dataset.Davis2016TL import get_inpainting_dataloader
from layer import fn
from layer.genesis import Genesis
from layer.helper import cb2b
from util import figureUtil
from util.logUtil import logger

# program init
label_set = {}
bce_loss = nn.BCELoss(reduction='mean')
itr_times, g_losses, h_losses, d_losses, h_d_losses, accuracies, hash_dists = [], [], [], [], [], [], []


def load_label_classes(data_path):
    classes = os.listdir(data_path)
    for c in classes:
        label_set[c] = 0
    num_classes = len(classes)
    return num_classes


def get_classes_label(label):
    l_set = label_set.copy()
    l_set[label] = 1
    return list(l_set.values())


def get_tensor_target(labels: []):
    ts = []
    for l in labels:
        ts.append(get_classes_label(l))
    x = np.asarray(ts, dtype=np.float32).repeat(3, axis=0)
    return torch.from_numpy(x)


def train(args_, dataloader_, test_loader_, num_classes):
    # init
    genesis = Genesis(224, GlobalConfig.PATCH_SIZE, args_.local_rank, [args_.local_rank])
    device = genesis.device

    # running
    test_itr = enumerate(test_loader_)
    idx = GlobalConfig.CHECKPOINT
    for epoch in range(1000):
        train_cache = TrainCache(size=16)
        _thread.start_new_thread(load_cache, (dataloader_, train_cache,))
        while not train_cache.finished:
            if train_cache.has_item():
                try:
                    _, item = train_cache.next_data()
                    train_step(genesis, item, idx, epoch, device)
                    test_step(genesis, idx, epoch, test_itr, device)
                except Exception as e:
                    print(e)
                    if isinstance(e, StopIteration):
                        test_itr = enumerate(test_loader_)
                idx += 1


def train_step(genesis: Genesis, item: TrainItem, idx, epoch, device):
    # HashNet
    src = cb2b(item.src, device)
    fake = cb2b(item.fake, device)
    masks = cb2b(item.mask, device)
    loss_g, g = train_g(genesis, [src, fake], masks, idx)
    # epoch log
    logger.info("Train Epoch:{}/{},G Loss:{:.5f}".format(epoch, idx, loss_g))


def test_step(genesis: Genesis, idx, epoch, test_itr, device):
    if idx % 100 == 0:
        genesis.eval()
        _, (label, _, _, _, sources, fakes, masks) = test_itr.__next__()
        # HashNet
        fakes = cb2b(fakes, device)
        sources = cb2b(sources, device)
        masks = cb2b(masks, device)
        g = genesis.g([sources, fakes])
        # save generate mask
        figureUtil.merge_pic(g, masks, 'images/{}_{}_{}_test.jpg'.format(epoch, idx, 0))
        # save generate mask
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
            g_losses.append(round(g_loss.item(), 3))
        return g_loss, g
    except Exception as e:
        print(e)


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r'Y:\vrf_')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--type', type=int, default=0)
if __name__ == '__main__':
    args = parser.parse_args()
    print('args:{}'.format(args))
    dataloader, test_loader, num_classes = None, None, 0
    if args.type == 0:
        dataloader = get_dataloader(set_path=os.path.join(args.path, GlobalConfig.TRAIN))
        test_loader = get_dataloader(mode=GlobalConfig.TEST,
                                     set_path=os.path.join(args.path, GlobalConfig.TEST),
                                     num_workers=0)
        num_classes = load_label_classes(os.path.join(args.path, GlobalConfig.TRAIN))
    elif args.type == 1:
        dataloader = get_inpainting_dataloader(set_path=os.path.join(args.path, GlobalConfig.TRAIN))
        test_loader = get_inpainting_dataloader(mode=GlobalConfig.TEST,
                                                set_path=os.path.join(args.path, GlobalConfig.TEST),
                                                num_workers=0)
        num_classes = load_label_classes(os.path.join(args.path, GlobalConfig.TRAIN, 'src'))
    train(args, dataloader, test_loader, num_classes)
