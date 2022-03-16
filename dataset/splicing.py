import os
import random

from dataset.Base import BaseVideoDataset, DataItem


class SplicingDataset(BaseVideoDataset):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def __getitem__(self, index):
        files, video_data = self.getitem(index)
        video_data: DataItem = video_data
        i = random.randint(-3, 100)
        fake_data = self.read_data(video_data.fake_dir, files, op=i)
        mask_data = self.read_data(video_data.mask_dir, files, op=i, mask=True)
        for i in range(len(files)):
            files[i] = os.path.join(video_data.fake_dir,files[i])
        return video_data.label, fake_data, mask_data, files

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
                    for _f in os.listdir(fake_):
                        mask = os.path.join(mask_, _f)
                        fake = os.path.join(fake_, _f)
                        data_item = DataItem(src, cls, start, mask, fake)
                        start = data_item.end
                        self.data.append(data_item)
            else:
                listdir = sorted(os.listdir(fake_dir))
                for _f in listdir:
                    label = _f
                    mask = os.path.join(mask_dir, _f)
                    fake = os.path.join(fake_dir, _f)
                    src = os.path.join(src_dir, _f)
                    data_item = DataItem(src, label, start, mask, fake)
                    start = data_item.end
                    self.data.append(data_item)
        self.count(start)

