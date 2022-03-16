import json
import os
import sys

import GlobalConfig
import torch
from PIL import Image
from torch import Tensor

from dataset.DFTL import get_dataloader
from dataset.Davis2016TL import get_inpainting_dataloader
from layer import helper
from layer.helper import tensor_to_binary, compute_hamming_dist, to_image
from layer.localizator import Localizator
from layer.vit_hash import ViTHash
from train_g import load_label_classes
from util.logUtil import logger

device = torch.device("cuda:0")
hashmap = {}


def test(data_type: int, path, test_op, hash_bits):
    dataloader, net_h, net_g, count, acc_sum = None, None, None, 0, 0
    if data_type == 0:
        dataloader = get_dataloader(set_path=os.path.join(path, GlobalConfig.TEST),
                                    batch_size=1,
                                    mode=GlobalConfig.TEST, test_op=test_op)
        num_classes = load_label_classes(os.path.join(path, GlobalConfig.TRAIN))
        net_h = ViTHash(hash_bits=hash_bits)
        net_h.load_state_dict(torch.load('./model/deepfake/' + str(hash_bits) + '_net_h.pth', map_location=device))
        load_map('./model/deepfake/' + str(hash_bits) + '_hash.json')
        # generator
        net_g = Localizator()
        net_g.load_state_dict(torch.load('./model/deepfake/net_g.pth', map_location=device))
    else:
        dataloader = get_inpainting_dataloader(set_path=os.path.join(path, GlobalConfig.TEST),
                                               batch_size=1,
                                               mode=GlobalConfig.TEST, test_op=test_op)
        num_classes = load_label_classes(os.path.join(path, GlobalConfig.TRAIN, 'src'))
        net_h = ViTHash(hash_bits=hash_bits)
        net_h.load_state_dict(torch.load('./model/inpainting/' + str(hash_bits) + '_net_h.pth', map_location=device))
        load_map('./model/inpainting/' + str(hash_bits) + '_hash.json')
        # generator
        net_g = Localizator()
        net_g.load_state_dict(torch.load('./model/inpainting/net_g.pth', map_location=device))
    net_h = net_h.to(device)
    net_h.eval()
    net_g = net_g.to(device)
    net_g.eval()
    for idx, (label, src_imgs, src_images, fake_images, src, fakes, masks) in enumerate(dataloader):
        fakes = helper.cb2b(fakes, device)
        # generator
        # test_g(data_type, fake_images, fakes, idx, masks, net_g, src, src_images, test_op)

        count += len(label)
        acc_sum = test_h(acc_sum, data_type, fakes, hash_bits, idx, label, net_h, path, src_imgs, test_op)
    logger.info("hash_bits:{},test_op:{},acc_sum :{}/{}, acc:{:.5f}".format(
        hash_bits, test_op, acc_sum, count, acc_sum / count))


def test_h(acc_sum, data_type, fakes, hash_bits, idx, label, net_h, path, src_imgs, test_op):
    h = net_h(fakes)
    acc, find_labels = find_index_test(h.clone().detach().cpu(), label, hash_bits)
    acc_sum += acc
    # hashing
    logger.info("Test :{}, hash_bits:{}, acc:{:.5f}".format(idx, hash_bits, acc))
    find_files = find_src_img(os.path.join(path, GlobalConfig.TRAIN), find_labels, _type=data_type)
    merge_pic_trace(src_imgs, find_files, 'test{}/images/{}_{}_trace.jpg'.format(test_op, idx, data_type))
    return acc_sum


def test_g(data_type, fake_images, fakes, idx, masks, net_g, src, src_images, test_op):
    src = helper.cb2b(src, device)
    masks = helper.cb2b(masks, device)
    g = net_g([src, fakes])
    merge_pic_test(g, masks, f'test{test_op}/images/{idx}_{data_type}_mask.jpg')


def find_index_test(hashset: Tensor, label: [], hash_bits):
    hashset = tensor_to_binary(hashset).numpy()
    real_count, find_labels, f_l = 0, [], None
    for i in range(len(label)):
        l_ = label[i]
        f_l = l_
        min_dis = hash_bits
        for k, v in hashmap.items():
            dis = compute_hamming_dist(v, hashset[i])
            if dis < min_dis:
                min_dis = dis
                f_l = k
        if l_ == f_l:
            real_count += 1
        else:
            logger.info(f'error:{l_}, {f_l}, {min_dis}')
        find_labels.append(f_l)
    return real_count, find_labels


def find_src_img(path, label: [], _type=0):
    files = []
    for l_ in label:
        path_f = os.path.join(path, l_, 'src') if _type == 0 else os.path.join(path, 'src', l_)
        file = os.listdir(path_f)[0]
        file = os.path.join(path_f, file)
        files.append(file)

    return files


def merge_pic(files: [], name):
    print(files)
    h, w, b = 224, 224, len(files)
    image = Image.open(files[0][0])
    images = Image.new(image.mode, (w * b, h))
    for i in range(b):
        image = files[i][0]
        img = Image.open(image)
        img = img.resize((224, 224))
        images.paste(img, box=(i * w, 0))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def merge_pic_trace(find_files: [], files: [], name):
    h, w, b = 224, 224, len(find_files),
    image = Image.open(files[0])
    images = Image.new(image.mode, (w * b, h * 2))
    for i in range(b * 2):
        if i % 2 == 0:
            image = find_files[i // 2]
        elif i % 2 == 1:
            image = files[i // 2]
        img = Image.open(image)
        img = img.resize((224, 224))
        images.paste(img, box=((i // 2) * w, (i % 2) * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def merge_pic_test(g_tensor: Tensor, mask: Tensor, name):
    b, t, h, w = g_tensor.shape
    g_tensor = g_tensor.detach().cpu()
    img = to_image(g_tensor[0][0])
    images = Image.new(img.mode, (w * t, h * b * 2))
    for j in range(b * 2):
        for i in range(t):
            if j % 2 == 1:
                img = g_tensor[j // 2][i]
            else:
                img = mask[j // 2][i]
            img = to_image(img)
            images.paste(img, box=(i * w, j * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def deepfake_dataset(set_path=r'/home/dell/soft/ppf/vrf/train'):
    items = sorted(os.listdir(set_path))
    images = None
    for i in range(121):
        item = items[i]
        item_path = os.path.join(set_path, item)
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            img = sorted(os.listdir(src_dir))[0]
            img = Image.open(os.path.join(src_dir, img))
            if images is None:
                images = Image.new(img.mode, (224 * 11, 224 * 11))
            img = img.resize((224, 224))
            images.paste(img, box=((i % 11) * 224, (i // 11) * 224))
    images.save('deepfake.jpg')


def inpainting_dataset(set_path=r'/home/dell/soft/ppf/inpainting/train'):
    set_path = os.path.join(set_path, 'src')
    items = sorted(os.listdir(set_path))
    images = None
    for i in range(49):
        item = items[i]
        item_path = os.path.join(set_path, item)
        if os.path.isdir(item_path):
            img = sorted(os.listdir(item_path))[0]
            img = Image.open(os.path.join(item_path, img))
            if images is None:
                images = Image.new(img.mode, (224 * 7, 224 * 7))
            img = img.resize((224, 224))
            images.paste(img, box=((i // 7) * 224, (i % 7) * 224))
    images.save('inpainting.jpg')


def load_map(file):
    try:
        if os.path.exists(file):
            with open(file, 'r') as f:
                content = json.load(f)
                hashmap.update(content)
        return True
    except BaseException as e:
        print(e)
        return False


if __name__ == '__main__':
    type_ = int(sys.argv[1])
    path_ = sys.argv[2]
    bit = int(sys.argv[3])
    for i in range(-1, 5):
        test(type_, path_, i, bit)
