import os
import random

import torch
from torch.utils import data as tud

from config import BaseConfig
from dataset.Base import BaseVideoDataset
from dataset.DFTL import DataItem


class Davis2016Dataset(BaseVideoDataset):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def _load_data(self):
        start = 0
        item_path = os.path.abspath(self.set_path)
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'mask')
            listdir = sorted(os.listdir(src_dir))
            fake_list = sorted(os.listdir(fake_dir))
            print(fake_list)
            for _f in listdir:
                label = _f
                mask = os.path.join(mask_dir, _f)
                src = os.path.join(src_dir, _f)
                fakes = []
                for fake_ in fake_list:
                    fake = os.path.join(fake_dir, fake_, _f)
                    fakes.append(fake)
                data_item = DataItem(src, fakes, mask, label, start)
                start = data_item.end
                self.data.append(data_item)
        self.length = start // self.batch_size

    def __getitem__(self, index):
        files, video_data = self.getitem(index)
        if self.mode == BaseConfig.TRAIN:
            video_data: DataItem = video_data
            i = random.randint(-3, 100)
            src = self.read_data(video_data.src_dir, files, op=i)
            hashes = [src]
            for _ in range(2):
                fake_idx = random.randint(0, 100) % len(video_data.fake_dir)
                fake_data = self.read_data(video_data.fake_dir[fake_idx], files, op=i)
                hashes.append(fake_data)
            mask_data = self.read_data(video_data.mask_dir, files, mask=True, op=i)
            for i in range(len(files)):
                files[i] = os.path.join(video_data.fake_dir, files[i])
            return video_data.label, hashes, mask_data
        else:
            src_files, fake_files = [], []
            for f in files:
                src_files.append(os.path.join(video_data.src_dir, f))
                fakes = []
                for fake_dir in video_data.fake_dir:
                    fakes.append(os.path.join(fake_dir, f))
                fake_files.append(fakes)
            mask_ = self.read_data(video_data.mask_dir, files, mask=True)
            src = self.read_data(video_data.src_dir, files)
            fakes = []
            for fake_dir in video_data.fake_dir:
                fake = self.read_data(fake_dir, files)
                fakes.append(fake)
            return video_data.label, src_files, fake_files, src, fakes, mask_
