import os

from torch.utils import data as tud

import GlobalConfig
from dataset.dataset import VrfDataset, DataItem


class InpaintingDataset(VrfDataset):
    def __init__(self, **kwargs):
        super(InpaintingDataset, self).__init__(**kwargs)

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
                for fake_ in fake_list:
                    fake = os.path.join(fake_dir, fake_, _f)
                    src = os.path.join(src_dir, _f)
                    data_item = DataItem(src, fake, mask, label, start)
                    start = data_item.end
                    self.data.append(data_item)
        self.length = start // self.batch_size


def get_inpainting_dataloader(set_path, mode=GlobalConfig.TRAIN,
                              num_workers=min(os.cpu_count(), GlobalConfig.BATCH_SIZE),
                              batch_size=GlobalConfig.BATCH_SIZE, test_op=-1, shuffle=True):
    dataset = InpaintingDataset(set_path=set_path, mode=mode, test_op=test_op, type=1)
    dataloader = tud.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
    return dataloader
