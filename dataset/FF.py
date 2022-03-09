import os
import random

import torch

from dataset.Base import BaseVideoDataset, DataItem, BaseTrainItem

compresses = ['raw', 'c23', 'c40']
trace_listdir = ['face2face', 'faceshifter', 'faceswap', 'deepfakes', 'neuraltextures']
mask_listdir = ['face2face', 'faceswap', 'deepfakes', 'neuraltextures']


class FFDataset(BaseVideoDataset):
    mask = False

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def __getitem__(self, index):
        files, video_data = self.getitem(index)
        if self.cfg.mode == self.cfg.TRAIN:
            video_data: DataItem = video_data
            i = random.randint(-3, 100)
            src = self.read_data(video_data.src_dir, files, op=i)
            hashes, masks = [src], []
            for _ in range(2):
                fake_idx = random.randint(0, 100) % len(video_data.fake_dir)
                fake_data = self.read_data(video_data.fake_dir[fake_idx], files, op=i)
                hashes.append(fake_data)
                if self.mask:
                    mask_data = self.read_data(video_data.mask_dir[fake_idx], files, mask=True, op=i)
                    masks.append(mask_data)
            return video_data.label, hashes, masks
        else:
            src_files, fake_files = [], []
            for f in files:
                src_files.append(os.path.join(video_data.src_dir, f))
                fakes = []
                for fake_dir in video_data.fake_dir:
                    fakes.append(os.path.join(fake_dir, f))
                fake_files.append(fakes)
            src = self.read_data(video_data.src_dir, files)
            fakes, masks = [], []
            for i in len(video_data.fake_dir):
                fake = self.read_data(video_data.fake_dir[i], files)
                fakes.append(fake)
                if self.mask:
                    mask = self.read_data(video_data.mask_dir[i], files)
                    masks.append(mask)
            return video_data.label, [src_files, fake_files, src, fakes], masks

    def _load_data(self):
        start = 0
        item_path = self.cfg.set_path
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'masks')
            fakes, masks = [], []
            for item in os.listdir(src_dir):
                listdir = mask_listdir if self.mask else trace_listdir
                src = os.path.join(src_dir, item)
                label = item
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
