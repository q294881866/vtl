import _thread
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from config import DFTLConfig, Davis2016Config, FFConfig, BaseConfig
from dataset.Base import TrainItem, get_dataloader, TrainCache, load_cache
from dataset.DFTL import DFTLDataset
from dataset.Davis2016TL import Davis2016Dataset
from dataset.FF import FFDataset
from layer import helper
from layer.fn import hash_triplet_loss
from layer.genesis import Genesis
from layer.helper import cb2b
from util.logUtil import logger

# program init
label_set = {}
bce_loss = nn.BCELoss(reduction='mean')
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


def train(cfg, dataloader_, test_loader_):
    # init
    genesis = Genesis(cfg, train_h=True)
    device = genesis.device

    # running
    test_itr = enumerate(test_loader_)
    for epoch in range(cfg.EPOCH):
        train_cache = TrainCache(size=32)
        _thread.start_new_thread(load_cache, (dataloader_, train_cache))
        while not train_cache.finished:
            if train_cache.has_item():
                idx, item = train_cache.next_data()
                train_step(genesis, item, idx, epoch, device)
                test_step(genesis, idx, epoch, test_itr, device)


def train_step(genesis: Genesis, item: TrainItem, idx, epoch, device):
    # HashNet
    hashes = cb2b(item.hashes, device)
    loss_h, loss_d = train_h(genesis, hashes, item.label, device)
    # epoch log
    logger.info(f"Train Epoch:{epoch}/{idx},H Loss:{loss_h.item():.5f}, D Loss:{loss_d.item():.5f},hash dis:{helper.hash_intra_dis():.5f}")


def test_step(genesis: Genesis, idx, epoch, test_itr, device):
    if idx % 100 == 0:
        genesis.eval()
        _, (label, _, _, fake, _) = test_itr.__next__()
        # HashNet
        fakes = cb2b(fake, device)
        h = genesis.h(fakes)
        acc = helper.find_index(h, label)
        # epoch log
        logger.info("Test :{}/{}, acc:{:.5f}".format(epoch, idx, acc))
        genesis.save(f'model/{genesis.cfg.type}_{epoch}_{idx}_')
        helper.save_hash(f'model/{genesis.cfg.type}_{epoch}_{idx}_', genesis.cfg.HASH_BITS)
        genesis.train()


def train_h(genesis: Genesis, train_data, label, device):
    # train
    d, h = genesis.h(train_data)
    h_loss = hash_triplet_loss(h, label, device)
    # d loss
    d_label = get_tensor_target(label).to(device)
    d_loss = bce_loss(d.flatten(), d_label.flatten())
    d_h_loss = d_loss + h_loss
    # backward
    genesis.reset_grad()
    d_h_loss.backward()
    genesis.opt_h.step()
    genesis.scheduler_h.step()
    return h_loss, d_loss


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

    if args_.type == 'DFTL':
        num_classes = load_label_classes(os.path.join(args_.path, BaseConfig.TRAIN))
    elif args_.type == 'Davis2016':
        num_classes = load_label_classes(os.path.join(args_.path, BaseConfig.TRAIN, 'src'))
    elif args_.type == 'FF':
        num_classes = load_label_classes(os.path.join(args_.path, BaseConfig.TRAIN, 'src'))
    train_cfg.NUM_CLASSES = num_classes
    train(train_cfg, dataloader, test_loader)
