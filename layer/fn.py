import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from layer import helper
from layer.helper import tensor_to_binary, to_hashmap


def hash_triplet_loss(hashset, label, device):
    hashset = tensor_to_binary(hashset)
    h_map = to_hashmap(hashset, label)
    c = len(hashset) // len(label)
    intra_loss, inter_loss = 0, 0
    for k, v in h_map.items():
        for i in range(len(label)):
            l_ = label[i]
            idx = i * c
            for j in range(c):
                _loss = torch.sub(torch.from_numpy(np.asarray(v)).to(device), hashset[idx + j]).abs().mean()
                if k != l_:
                    inter_loss = (inter_loss + 1 - _loss) / 2
                if k == l_:
                    intra_loss = (intra_loss + _loss) / 2
    helper.update_hash(h_map)
    return inter_loss + intra_loss


def global_hash_loss(hashset, label, device):
    hashset = tensor_to_binary(hashset)
    intra_loss, inter_loss = 0, 0
    c = len(hashset) // len(label)
    for i in range(len(label)):
        l_ = label[i]
        idx = i * c
        for j in range(c):
            for k_, v_ in helper.get_hashmap().items():
                _loss = torch.sub(torch.from_numpy(np.asarray(v_)).to(device), hashset[idx + j]).abs().mean()
                if k_ == l_:
                    intra_loss = (intra_loss + _loss) / 2
    return intra_loss


def hash_loss(hashset, label, device):
    return hash_triplet_loss(hashset, label, device) + global_hash_loss(hashset, label, device)


def mask_loss(input_: Tensor, target: Tensor):
    return torch.sub(input_, target.clone().detach()).abs().mean()
