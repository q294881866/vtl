import os
import random

import torch

from dataset.Base import BaseVideoDataset, DataItem

compresses = ['raw', 'c23', 'c40']
trace_listdir = ['face2face', 'faceshifter', 'faceswap', 'deepfakes', 'neuraltextures']
# trace_listdir = ['deepfakes']
mask_listdir = ['face2face', 'faceswap', 'deepfakes', 'neuraltextures']


class FFDataset(BaseVideoDataset):
    mask = False

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
        item_path = self.cfg.set_path
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'masks')
            for item in os.listdir(src_dir):
                listdir = mask_listdir if self.mask else trace_listdir
                src = os.path.join(src_dir, item)
                label = item
                fakes, masks = [], []
                for cls in listdir:
                    fake = os.path.join(fake_dir, cls, compresses[0], item)
                    fakes.append(fake)
                    if self.mask:
                        mask = os.path.join(mask_dir, cls, item)
                        masks.append(mask)
                data_item = DataItem(src, label, start, masks, fakes)
                start = data_item.end
                self.data.append(data_item)
        self.count(start)
