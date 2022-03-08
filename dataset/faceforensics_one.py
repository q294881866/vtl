import os
import random

import torch

from config import FFConfig
from dataset.Base import BaseVideoDataset, DataItem, BaseTrainItem

listdir = ['face2face', 'faceshifter', 'faceswap', 'deepfakes', 'neuraltextures']


class FFDataset(BaseVideoDataset):
    def __init__(self, cfg):
        super().__init__(cfg=FFConfig)

    def __getitem__(self, index):
        files, video_data = self.getitem(index)
        if self.mode == self.cfg.TRAIN:
            video_data: DataItem = video_data
            i = random.randint(-3, 100)
            src = self.read_data(video_data.src_dir, files, op=i)
            hashes = [src, self.read_data(video_data.fake_dir[self.cfg.choice_idx], files)]
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

    def _load_data(self):
        start = 0
        item_path = self.cfg.set_path
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'mask')
            for item in os.listdir(src_dir):
                src = os.path.join(src_dir, item)
                label = item
                for cls in listdir:
                    mask = os.path.join(mask_dir, cls, item)
                    fake_compress = os.path.join(fake_dir, cls)
                    fake_dirs = []
                    for fake_c in os.listdir(fake_compress):
                        fake = os.path.join(fake_compress, fake_c, item)
                        fake_dirs.append(fake)
                    data_item = DataItem(src, label, start, mask, fake_dirs)
                    start = data_item.end
                    self.data.append(data_item)
        self.count(start)
