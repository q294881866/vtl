import os
import random
import time
from abc import ABCMeta

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils import data as tud
from torch.utils.data import Dataset

from layer.helper import tensor_resize, tensor2img
from util.logUtil import logger


class BaseTrainItem:
    idx = None

    def __init__(self):
        super().__init__()
        pass


class TrainItem(BaseTrainItem):
    idx = 0

    def __init__(self, label, hashes, mask):
        super(TrainItem, self).__init__()
        self.label = label
        self.hashes = hashes
        self.mask = mask


class TrainCache:
    def __init__(self, size):
        self.cache = {}
        self.size = size
        self.finished = False

    def put(self, idx, item: BaseTrainItem):
        self.cache[idx] = item
        item.idx = idx

    def is_stop(self):
        return len(self.cache) > self.size

    def next_data(self):
        return self.cache.popitem()

    def has_item(self):
        return len(self.cache) > 0

    def finish(self):
        self.finished = True


class DataItem(object):
    def __init__(self, src, label, start, mask=None, fake=None):
        self.label = label
        self.start = start
        self.src_dir = src
        self.mask_dir = mask
        self.fake_dir = fake
        self.files = sorted(os.listdir(self.src_dir))
        length_files = len(self.files)
        if isinstance(self.fake_dir, list):
            for fake_dir in self.fake_dir:
                files = sorted(os.listdir(fake_dir))
                if len(files) < length_files:
                    length_files = len(files)
                    self.files = files
        self.end = start + length_files


class BaseVideoDataset(Dataset, metaclass=ABCMeta):
    def __len__(self):
        return self.length

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data = []
        self._load_data()
        logger.info(f'Dataset,loading.{self.length}-{self.cfg.NUM_FRAMES}-{self.cfg.BATCH_SIZE}-{cfg}')

    def _load_data(self):
        pass

    def count(self, count):
        self.length = count // self.cfg.FRAMES_STEP

    def _get_files(self, idx):
        video_data, start, end, size = None, 0, 0, self.cfg.NUM_FRAMES
        for e in self.data:
            video_data: DataItem = e
            # global length
            start = idx * self.cfg.FRAMES_STEP
            if start < video_data.end:
                end = start + size
                if end > video_data.end:
                    # item length
                    start = video_data.end - size
                    end = video_data.end
                break
        start = start - video_data.start
        end = end - video_data.start
        files = video_data.files[start:end]
        return files, video_data

    def getitem(self, index):
        files, video_data = self._get_files(index)
        while True:
            try:
                assert len(files) == self.cfg.NUM_FRAMES, 'Inconsistent data length'
                self.read_data(video_data.src_dir, files)
                return files, video_data
            except BaseException as e:
                logger.error('dir:{} ERROR:{}'.format(video_data.src_dir, e))
                index = random.randint(0, self.length)
                files, video_data = self._get_files(index)
                continue

    def read_data(self, _dir, files, op=0, mask=False):
        tensors = []
        if self.cfg.mode == self.cfg.TEST:
            op = 0
        for f in files:
            _f = os.path.join(_dir, f)
            im = Image.open(_f)
            if op % 2 == 1:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if mask:
                tensor = to_mask_tensor(im, self.cfg.IMAGE_SIZE).unsqueeze(0)
            else:
                if op % 6 == 5:
                    im = im.filter(ImageFilter.DETAIL)
                elif op % 6 == 1:
                    im = im.filter(ImageFilter.GaussianBlur)
                elif op % 6 == 2:
                    im = im.filter(ImageFilter.BLUR)
                elif op % 6 == 3:
                    im = im.filter(ImageFilter.MedianFilter)
                tensor = tensor_resize(self.cfg.loader(im).unsqueeze(0), self.cfg.IMAGE_SIZE)
            tensors.append(tensor)
        data = torch.cat(tensors, dim=0)
        if self.cfg.image_based:
            data = torch.squeeze(data, dim=0)
        return data.unsqueeze(0)


def get_dataloader(dataset, cfg):
    if cfg.IS_DISTRIBUTION:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.cuda.device_count(),
            rank=0,
            shuffle=True
        )
    else:
        train_sampler = None
    dataloader = tud.DataLoader(dataset=dataset,
                                num_workers=min(os.cpu_count(), cfg.BATCH_SIZE),
                                batch_size=cfg.BATCH_SIZE, shuffle=cfg.shuffle,
                                sampler=train_sampler,
                                )
    return dataloader


def load_cache(dataloader, train_cache: TrainCache):
    for values in enumerate(dataloader):
        cache = TrainItem(*values[1])
        train_cache.put(values[0], cache)
        while train_cache.is_stop():
            time.sleep(1)
    time.sleep(10)
    train_cache.finish()


def to_mask_tensor(img, image_size=224):
    img = img.convert('L')
    img = img.resize((image_size, image_size))
    img = np.asarray(img, dtype=np.int32)
    mask = torch.zeros([image_size, image_size])
    for h in range(image_size):
        for w in range(image_size):
            if img[h, w] > 120:
                mask[h, w] = 1.0
    return torch.unsqueeze(mask, 0)


if __name__ == '__main__':
    im = Image.open('../1.png')
    im = to_mask_tensor(im)
    im = tensor2img(im)
    im.show()
