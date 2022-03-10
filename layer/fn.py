import numpy as np
import torch
from torch import Tensor

from layer import helper
from layer.helper import to_hashmap


def hash_triplet_loss(hashset, label, device):
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
    multi = inter_loss.item() // intra_loss.item() // 10
    return inter_loss + intra_loss * multi * 10


def mask_loss(input_: Tensor, target: Tensor):
    return torch.sub(input_, target.clone().detach()).abs().mean()
