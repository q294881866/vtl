import _thread
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from config import DFTLConfig, Davis2016Config, FFConfig, BaseConfig
from dataset.Base import TrainCache, load_cache, TrainItem, get_dataloader
from dataset.DFTL import DFTLDataset
from dataset.Davis2016TL import Davis2016Dataset
from dataset.FF import FFDataset
from layer import helper
from layer.fn import hash_triplet_loss
from layer.genesis import Genesis
from layer.helper import cb2b
from util import figureUtil
from util.logUtil import logger

# program init
label_set = {}
bce_loss = nn.BCELoss(reduction='mean')
itr_times, g_losses, h_losses, d_losses, h_d_losses, accuracies, hash_dists = [], [], [], [], [], [], []
choices = {
    'DFTL': (DFTLDataset, DFTLConfig),
    'Davis2016': (Davis2016Dataset, Davis2016Config),
    'FF': (FFDataset, FFConfig),
}


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


def train(cfg: BaseConfig, dataloader_, test_loader_):
    # init
    genesis = Genesis(cfg, train_h=True)
    device = genesis.device

    # running
    test_itr = enumerate(test_loader_)
    for epoch in range(cfg.EPOCH):
        train_cache = TrainCache(size=16)
        _thread.start_new_thread(load_cache, (dataloader_, train_cache, TrainItem))
        while not train_cache.finished:
            if train_cache.has_item():
                try:
                    idx, item = train_cache.next_data()
                    train_step(genesis, item, idx, epoch, device)
                    test_step(genesis, idx, epoch, test_itr, device)
                except Exception as e:
                    print(e)
                    test_itr = enumerate(test_loader_)
                idx += 1
        path = './images/' + str(epoch)
        figureUtil.analyze_loss(path + '_loss.jpg', itr_times, h_losses, d_losses, accuracies)
        figureUtil.analyze_hash_dist(path + '_acc.jpg', itr_times, hash_dists)


def train_step(genesis: Genesis, item: TrainItem, idx, epoch, device):
    # HashNet
    hashes = cb2b(item.hashes, device)
    loss_h, acc = train_h(genesis, hashes, item.label, device, idx)
    # epoch log
    logger.info("Train Epoch:{}/{},H Loss:{:.5f},hash dis:{:.5f} acc:{:.5f}".
                format(epoch, idx, loss_h, helper.hash_intra_dis(), acc))


def test_step(genesis: Genesis, idx, epoch, test_itr, device):
    if idx % 100 == 0:
        genesis.eval()
        _, (label, _, _, _, sources, fakes, masks) = test_itr.__next__()
        # HashNet
        fakes = cb2b(fakes, device)
        h = genesis.h(fakes)
        acc = helper.find_index(h, label)
        # epoch log
        logger.info("Test :{}/{}, acc:{:.5f}".format(epoch, idx, acc))
        genesis.save('model/{}_{}_'.format(epoch, idx))
        helper.save_hash('model/{}_{}_'.format(epoch, idx), genesis.cfg.HASH_BITS)
        genesis.train()


def train_h(genesis: Genesis, train_data, label, device, idx):
    # train
    d, h = genesis.h(train_data)
    h_loss = hash_triplet_loss(h, label, d)
    # backward
    genesis.reset_grad()
    h_loss.backward()
    genesis.opt_h.step()
    # genesis.scheduler_h.step()
    # hashcode accuracy
    acc = helper.find_index(h, label)
    if idx % 100 == 0:
        itr_times.append(idx)
        h_losses.append(round(h_loss.item(), 3))
        accuracies.append(round(acc, 3))
        hash_dists.append(round(helper.hash_intra_dis(), 3))
    return h_loss, acc


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r'Y:\vrf_')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--type', type=str, default='FF')
parser.add_argument('--bits', type=int, default=512)
if __name__ == '__main__':
    args_ = parser.parse_args()
    print('args:{}'.format(args_))
    helper.set_hash_bits(args_.bits)
    Dataset, cfg = choices[args_.type]

    train_cfg = cfg(cfg.TRAIN, os.path.join(args_.path, cfg.TRAIN), args_.path, args_.local_rank)
    dataset = Dataset(cfg=train_cfg)
    dataloader = get_dataloader(dataset=dataset, cfg=train_cfg)

    test_cfg = cfg(cfg.TEST, os.path.join(args_.path, cfg.TEST), args_.path, args_.local_rank)
    dataset = Dataset(cfg=test_cfg)
    test_loader = get_dataloader(dataset=dataset, cfg=test_cfg)

    if args_.type == 'DFTL':
        num_classes = load_label_classes(os.path.join(args_.path, BaseConfig.TRAIN))
    elif args_.type == 'Davis2016':
        num_classes = load_label_classes(os.path.join(args_.path, BaseConfig.TRAIN, 'src'))
    elif args_.type == 'FF':
        num_classes = load_label_classes(os.path.join(args_.path, BaseConfig.TRAIN, 'src'))
    train(train_cfg, dataloader, test_loader)
