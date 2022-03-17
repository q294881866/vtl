import os
import random

import torch

from dataset.Base import BaseVideoDataset, DataItem


class SplicingDataset(BaseVideoDataset):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

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

    def _load_data(self):
        start = 0
        item_path = os.path.abspath(self.cfg.set_path)
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'mask')
            if self.cfg.mode == self.cfg.TRAIN:
                listdir = sorted(os.listdir(fake_dir))
                for cls in listdir:
                    mask_ = os.path.join(mask_dir, cls)
                    fake_ = os.path.join(fake_dir, cls)
                    src = os.path.join(src_dir, cls)
                    fakes, masks = [], []
                    for _f in os.listdir(fake_):
                        mask = os.path.join(mask_, _f)
                        masks.append(mask)
                        fake = os.path.join(fake_, _f)
                        fakes.append(fake)
                    data_item = DataItem(src, cls, start, masks, fakes)
                    start = data_item.end
                    self.data.append(data_item)
            else:
                listdir = sorted(os.listdir(fake_dir))
                for _f in listdir:
                    label = _f
                    mask = os.path.join(mask_dir, _f)
                    fake = os.path.join(fake_dir, _f)
                    src = os.path.join(src_dir, _f)
                    data_item = DataItem(src, label, start, [mask], [fake])
                    start = data_item.end
                    self.data.append(data_item)
        self.count(start)
