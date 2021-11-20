import os
import random
import time

import numpy as np
import torch
from PIL import Image as Image, ImageFilter
from einops import rearrange
from torch.utils import data as tud
from torch.utils.data import Dataset

from . import GlobalConfig
from .helper import loader
from .logUtil import logger


class DataItem(object):
    def __init__(self, src, fake, mask, label, start, sub_dir):
        self.label = label
        self.sub_dir = sub_dir
        self.fake_dir = fake
        self.mask_dir = mask
        self.src_dir = src
        self.files = sorted(os.listdir(self.src_dir))
        self.end = start + len(self.files)
        self.start = start


class VrfDataset(Dataset):
    def __len__(self):
        return self.length

    def __init__(self, **kwargs):
        super(VrfDataset, self).__init__()
        # 数据指针，k:label,v:start index
        self.set_path = os.path.abspath(kwargs.get('set_path', GlobalConfig.SET_PATH))
        self.batch_size = kwargs.get('batch_size', GlobalConfig.NUM_FRAME)
        self.mode = kwargs.get('mode', GlobalConfig.TRAIN)
        self.test_op = kwargs.get('test_op', -1)
        self.type = kwargs.get('type', 0)

        self.data = []
        self._load_data()

    def _load_data(self):
        start, label = 0, -1
        listdir = sorted(os.listdir(self.set_path))
        for item in listdir:
            label += 1
            item_path = os.path.join(self.set_path, item)
            if os.path.isdir(item_path):
                src_dir = os.path.join(item_path, 'src')
                fake_dir = os.path.join(item_path, 'fake')
                mask_dir = os.path.join(item_path, 'mask')
                listdir = sorted(os.listdir(fake_dir))
                for _f in listdir:
                    fake = os.path.join(fake_dir, _f)
                    mask = os.path.join(mask_dir, _f)
                    data_item = DataItem(src_dir, fake, mask, label, start, item)
                    start = data_item.end
                    self.data.append(data_item)

        self.length = start // self.batch_size

    def _get_files(self, idx):
        video_data, start, end = None, 0, 0
        for e in self.data:
            video_data: DataItem = e
            # global length
            start = idx * GlobalConfig.NUM_FRAME
            if start < video_data.end:
                end = start + GlobalConfig.NUM_FRAME
                if end > video_data.end:
                    # item length
                    start = video_data.end - GlobalConfig.NUM_FRAME
                    end = video_data.end
                break
        start = start - video_data.start
        end = end - video_data.start
        files = video_data.files[start:end]
        return files, video_data

    def __getitem__(self, index):
        files, video_data = self._get_files(index)
        if self.mode == GlobalConfig.TRAIN:
            i = random.randint(-3, 5)
            if i % 3 == 0:
                target = read_data(video_data.src_dir, files, op=i)
            else:
                target = read_data(video_data.fake_dir, files, op=i)
            target = rearrange(target, "t c h w -> c t h w")
            return target, video_data.label
        else:
            # inpainting dataset, src_file for hash retrieval comparative experiment
            src_file = os.path.join(video_data.fake_dir, '00000.jpg')
            if self.type == 0:
                # human dataset
                src_file = os.path.join(video_data.fake_dir, files[0])
            target = read_data(video_data.fake_dir, files, op=self.test_op)
            target = rearrange(target, "t c h w -> c t h w")
            return target, video_data.label, src_file, video_data.sub_dir


def to_mask_tensor(img):
    img = np.asarray(img, dtype=np.int32)
    mask = torch.zeros([224, 224])
    for h in range(224):
        for w in range(224):
            for c in range(3):
                if img[h, w, c] > 180:
                    mask[h, w] = 1
    return mask


def read_data(_dir, files, op=-1, mask=False):
    try:
        tensors = None
        for f in files:
            _f = os.path.join(_dir, f)
            im = Image.open(_f)
            w, h = im.size
            if op == 4:
                x, y = int(w * 0.05), int(h * 0.05)
                x_, y_ = int(w * 0.95), int(h * 0.95)
                im = im.crop((x, y, x_, y_))
            if op == 0:
                im = im.filter(ImageFilter.DETAIL)
            elif op == 1:
                im = im.filter(ImageFilter.GaussianBlur)
            elif op == 2:
                im = im.filter(ImageFilter.BLUR)
            elif op == 3:
                im = im.filter(ImageFilter.MedianFilter)
            img = im.resize((224, 224))
            if mask:
                tensor = to_mask_tensor(img).unsqueeze(0)
            else:
                tensor = loader(img).unsqueeze(0)
            if tensors is None:
                tensors = tensor
            else:
                tensors = torch.cat([tensors, tensor], dim=0)
        return tensors
    except BaseException as e:
        logger.info('data read error:{},{},{}'.format(_dir, files, e))


class TrainItem:
    def __init__(self, label, hashes, src, fake, masks):
        super(TrainItem, self).__init__()
        self.label = label
        self.hashes = hashes
        self.src = src
        self.fake = fake
        self.masks = masks


class TrainCache:
    def __init__(self, size):
        self.cache = {}
        self.size = size
        self.finished = False

    def put(self, idx, item: TrainItem):
        self.cache[idx] = item

    def is_stop(self):
        return len(self.cache) > self.size

    def next_data(self):
        return self.cache.popitem()

    def has_item(self):
        return len(self.cache) > 0

    def finish(self):
        self.finished = True


def load_cache(dataloader, train_cache: TrainCache):
    # train cache
    for idx, (label, hashes, src, fake, masks) in enumerate(dataloader):
        cache = TrainItem(label, hashes, src, fake, masks)
        train_cache.put(idx, cache)
        while train_cache.is_stop():
            time.sleep(1)
    time.sleep(10)
    train_cache.finish()


def get_dataloader(set_path, mode=GlobalConfig.TRAIN, num_workers=min(os.cpu_count(), GlobalConfig.BATCH_SIZE),
                   batch_size=GlobalConfig.BATCH_SIZE, test_op=-1, shuffle=True):
    dataset = VrfDataset(set_path=set_path, mode=mode, test_op=test_op)
    dataloader = tud.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
    return dataloader
