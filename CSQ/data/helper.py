import json
import os
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torchvision.transforms import transforms

from . import GlobalConfig

loader = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.565, 0.556, 0.547],
                         std=[0.232, 0.233, 0.234])
])
unloader = transforms.ToPILImage()
hashmap = {}


def set_hash_bits(hb):
    GlobalConfig.HASH_BIT = hb


def get_hash_bits():
    return GlobalConfig.HASH_BIT


def get_net_h(hash_bit=get_hash_bits()):
    return str(hash_bit) + '_net_h.pth'


def get_hash_json(hash_bit=get_hash_bits()):
    return str(hash_bit) + '_hash.json'


def gen_hash(hashset: []):
    """
    hash list to hash by select most common element
    Args:
        hashset: must be numpy array

    Returns:

    """
    arr = np.array(hashset)
    _hash = []
    for i in range(get_hash_bits()):
        bit_ = Counter(arr[:, i]).most_common(1)
        _hash.append(int(bit_[0][0]))
    return _hash


def to_binary(code):
    hashset = []
    code = np.asarray(code)
    for h in code:
        hash_code = []
        mean_v = h.mean()
        for i in h:
            if i > mean_v:
                hash_code.append(1)
            else:
                hash_code.append(0)
        hashset.append(hash_code)
    return hashset


def update_hash(h_map: {}):
    hash_bits = get_hash_bits()
    for k, v in h_map.items():
        hashmap[k] = v
        hash_bits = len(v)
    save_hash(hash_bits=hash_bits)


def get_hash(label):
    load_hash()
    if not hashmap.__contains__(label):
        hashmap[label] = np.random.randint(0, 2, get_hash_bits()).tolist()
    return hashmap[label]


def to_image(image):
    return unloader(image)


def save_hash(prefix='./', hash_bits=get_hash_bits()):
    _save(hashmap, prefix + get_hash_json(hash_bits))


def _save(map_, file):
    while not __save(map_, file):
        time.sleep(2)


def __save(map_, file):
    try:
        with open(file, 'w') as f:
            json.dump(map_, f)
        return True
    except BaseException as e:
        print(e)
        return False


def _load(map_, file):
    while not __load(map_, file):
        time.sleep(2)


def __load(map_, file):
    try:
        if os.path.exists(file):
            with open(file, 'r') as f:
                content = json.load(f)
                map_.update(content)
        return True
    except BaseException as e:
        print(e)
        return False


def get_hashmap():
    load_hash()
    return hashmap


def load_hash(file=get_hash_json()):
    _load(hashmap, file)


def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hash_intra_dis():
    size = len(hashmap)
    intra_dis = 0
    for arr1 in hashmap.values():
        for arr2 in hashmap.values():
            if arr1 != arr2:
                intra_dis = intra_dis + compute_hamming_dist(arr1, arr2)
    return intra_dis / (size * (size - 1))


def hashmap_mean():
    np_arr = np.array([list(item) for item in hashmap.values()])
    return np_arr.mean()


def tensor_to_binary(x: Tensor):
    return (torch.sign(x) + 1) / 2


def compute_hamming_dist(a, b):
    """
    Computes hamming distance vector wisely
    Args:
        a: row-ordered codes {0,1}
        b: row-ordered codes {0,1}
    Returns:
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.abs(a - b))


def find_index(hashset: Tensor, label: []):
    """
    Save generate images all in one
    Args:
        label: data label
        hashset: generate hashcode
        name: save name

    Returns:

    """
    hashset_ = tensor_to_binary(hashset)
    h_map = to_hashmap(hashset_, label)
    real_count = 0
    for k_, v_ in h_map.items():
        f_l = k_
        min_dis = get_hash_bits()
        for k, v in hashmap.items():
            dis = compute_hamming_dist(v, v_)
            if dis < min_dis:
                min_dis = dis
                f_l = k
        if k_ == f_l:
            real_count += 1
    return real_count / len(label)


def to_hashmap(hashset: Tensor, label):
    hashset_ = rearrange(hashset.detach().clone().cpu(), '(c b) ... -> c b ...', c=len(label))
    h_map = {}  # init
    for l_ in label:
        h_map[l_] = []
    for i in range(len(label)):
        l_ = label[i]
        h_map[l_].append(gen_hash(hashset_[i].numpy()))
    for k, v in h_map.items():
        h_map[k] = gen_hash(v)
    return h_map


def cb2b(x: Tensor, device):
    return rearrange(x, 'c b ... -> (c b) ...').to(device)
