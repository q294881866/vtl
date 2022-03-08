import os
import random
import time

import numpy as np
import torch
from PIL import Image as Image, ImageFilter
from torch.utils import data as tud
from torch.utils.data import Dataset

from config import BaseConfig
from dataset.Base import BaseVideoDataset
from layer.helper import loader
from util.logUtil import logger


class DataItem(object):
    def __init__(self, src, fake, mask, label, start):
        self.label = label
        self.fake_dir = fake
        self.mask_dir = mask
        self.src_dir = src
        self.files = sorted(os.listdir(self.src_dir))
        self.end = start + len(self.files)
        self.start = start


class DFTLDataset(BaseVideoDataset):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def _load_data(self):
        start = 0
        listdir = os.listdir(self.set_path)
        for item in listdir:
            item_path = os.path.join(self.set_path, item)
            if os.path.isdir(item_path):
                src_dir = os.path.join(item_path, 'src')
                label = os.path.basename(item_path)
                fake_dir = os.path.join(item_path, 'fake')
                mask_dir = os.path.join(item_path, 'mask')
                listdir = sorted(os.listdir(fake_dir))
                for _f in listdir:
                    fake = os.path.join(fake_dir, _f)
                    mask = os.path.join(mask_dir, _f)
                    data_item = DataItem(src_dir, fake, mask, label, start)
                    start = data_item.end
                    self.data.append(data_item)
        self.length = start // self.batch_size

    def _get_files(self, idx):
        video_data, start, end = None, 0, 0
        for e in self.data:
            video_data: DataItem = e
            # global length
            start = idx * BaseConfig.NUM_FRAMES
            if start < video_data.end:
                end = start + BaseConfig.NUM_FRAMES
                if end > video_data.end:
                    # item length
                    start = video_data.end - BaseConfig.NUM_FRAMES
                    end = video_data.end
                break
        start = start - video_data.start
        end = end - video_data.start
        files = video_data.files[start:end]
        return files, video_data

    def __getitem__(self, index):
        files, video_data = self._get_files(index)
        if self.mode == BaseConfig.TRAIN:
            i = random.randint(-3, 5)
            # 1 mask
            mask_ = self.read_data(video_data.mask_dir, files, mask=True, op=i)
            # 2 source
            src = self.read_data(video_data.src_dir, files, op=i)
            # 3. generates = source video + fake video
            fake = self.read_data(video_data.fake_dir, files)
            fake_ = self.read_data(video_data.fake_dir, files, op=i)
            hashes = torch.cat([src, fake, fake_], dim=0)
            return video_data.label, hashes, mask_
        else:
            # inpainting dataset, src_file for hash retrieval comparative experiment
            src_files, fake_files = [], []
            for f in files:
                src_files.append(os.path.join(video_data.src_dir, f))
                fake_files.append(os.path.join(video_data.fake_dir, f))
            mask_ = self.read_data(video_data.mask_dir, files, mask=True)
            src = self.read_data(video_data.src_dir, files)
            fake = self.read_data(video_data.fake_dir, files)
            return video_data.label, [src_files, fake_files, src, fake], mask_
