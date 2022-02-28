import os
import random

from dataset.Base import BaseVideoDataset, DataItem, BaseTrainItem


class FFDataset(BaseVideoDataset):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def __getitem__(self, index):
        files, video_data = self.getitem(index)
        video_data: DataItem = video_data
        i = random.randint(-3, 100)
        fake_data = self.read_data(video_data.fake_dir, files, op=i)
        for i in range(len(files)):
            files[i] = os.path.join(video_data.fake_dir, files[i])
        return video_data.label, fake_data, files

    def _load_data(self):
        start = 0
        item_path = self.cfg.set_path
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'mask')
            for item in os.listdir(src_dir):
                listdir = sorted(os.listdir(fake_dir))
                src = os.path.join(src_dir, item)
                label = item
                for _f in listdir:
                    fake = os.path.join(fake_dir, _f, item)
                    mask = os.path.join(mask_dir, _f, item)
                data_item = DataItem(src, label, start, mask, fake)
                start = data_item.end
                self.data.append(data_item)
        self.count(start)


class FFTrainItem(BaseTrainItem):
    def __init__(self, label, fake_data, files):
        super().__init__()
        self.label = label
        self.fake_data = fake_data
        self.files = files
