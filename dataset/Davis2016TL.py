import os
import random

import torch

from dataset.Base import BaseVideoDataset
from dataset.DFTL import DataItem


class Davis2016Dataset(BaseVideoDataset):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def _load_data(self):
        start = 0
        item_path = os.path.abspath(self.cfg.set_path)
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
        self.count(start)

    def __getitem__(self, index):
        files, video_data = self.getitem(index)
        i = random.randint(-3, 100)
        src = self.read_data(video_data.src_dir, files, op=i)
        if self.cfg.train_h:
            video_data: DataItem = video_data
            hashes = [src]
            for j in range(2):
                fake_idx = random.randint(0, 100) % len(video_data.fake_dir)
                fake_data = self.read_data(video_data.fake_dir[fake_idx], files, op=i)
                hashes.append(fake_data)
            return video_data.label, torch.cat(hashes, dim=0), hashes[1]
        else:
            idx = random.randint(0, 100) % len(video_data.fake_dir)
            fake_dir = video_data.fake_dir[idx]
            mask = self.read_data(video_data.mask_dir[idx], files, mask=True, op=i)
            fake = self.read_data(fake_dir, files, op=i)
            return src, fake, mask
