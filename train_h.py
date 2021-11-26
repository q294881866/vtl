import _thread
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import GlobalConfig
from dataset.dataset import TrainCache, load_cache, get_dataloader, TrainItem
from dataset.inpainting_dataset import get_inpainting_dataloader
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


def train(args_, dataloader_, test_loader_, num_classes, hash_bits):
    # init
    genesis = Genesis(224, GlobalConfig.PATCH_SIZE, args_.local_rank, [args_.local_rank],
                      num_classes=num_classes,
                      hash_bits=hash_bits,
                      data_type=args_.type,
                      train_h=True)
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
                    test_itr = enumerate(test_loader_)
                idx += 1
        path = './images/' + str(epoch)
        figureUtil.analyze_loss(path + '_loss.jpg', itr_times, h_losses, d_losses, accuracies)
        figureUtil.analyze_hash_dist(path + '_acc.jpg', itr_times, hash_dists)


def train_step(genesis: Genesis, item: TrainItem, idx, epoch, device):
    # HashNet
    hashes = cb2b(item.hashes, device)
    loss_h, loss_d, acc = train_h(genesis, hashes, item.label, device, idx)
    # epoch log
    logger.info("Train Epoch:{}/{},H Loss:{:.5f},D Loss:{:.5f},hash dis:{:.5f} acc:{:.5f}".
                format(epoch, idx, loss_h, loss_d, helper.hash_intra_dis(), acc))


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
        helper.save_hash('model/{}_{}_'.format(epoch, idx), genesis.hash_bits)
        genesis.train()


def train_h(genesis: Genesis, train_data, label, device, idx):
    # train
    d, h = genesis.h(train_data)
    h_loss = hash_triplet_loss(h, label, d)
    # d loss
    d_label = get_tensor_target(label).to(device)
    d_loss = bce_loss(d.flatten(), d_label.flatten())
    d_h_loss = h_loss + d_loss
    # backward
    genesis.reset_grad()
    d_h_loss.backward()
    genesis.opt_h.step()
    # genesis.scheduler_h.step()
    # hashcode accuracy
    acc = helper.find_index(h, label)
    if idx % 100 == 0:
        itr_times.append(idx)
        h_losses.append(round(h_loss.item(), 3))
        d_losses.append(round(d_loss.item(), 3))
        accuracies.append(round(acc, 3))
        hash_dists.append(round(helper.hash_intra_dis(), 3))
    return h_loss, d_loss, acc


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r'Y:\vrf_')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--type', type=int, default=0)
parser.add_argument('--bits', type=int, default=GlobalConfig.HASH_BIT)
if __name__ == '__main__':
    args = parser.parse_args()
    print('args:{}'.format(args))
    helper.set_hash_bits(args.bits)
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
    train(args, dataloader, test_loader, num_classes, args.bits)
