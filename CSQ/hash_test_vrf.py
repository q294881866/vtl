import argparse
import logging
import os

import torch.nn.parallel
from PIL import Image

import dataset
from data import GlobalConfig
from data import iterator_factory
from network.symbol_builder import get_symbol

criterion = torch.nn.BCELoss().cuda()


def predict_similar(model, data_loader, center, test_path, data_type):  # data_loader is database_loader or test_loader
    model.eval()
    label_map, idx_map = {}, {}
    path = os.path.join(test_path, GlobalConfig.TRAIN)

    def load_label_map(path_):
        idx = 0
        list_dir = os.listdir(path_) if data_type == 0 else os.listdir(os.path.join(path_, 'src'))
        for label in sorted(list_dir):
            label_map[label] = idx
            idx_map[idx] = label
            idx += 1

    load_label_map(path_=path)
    logging.info(idx_map)
    acc_sum, count = 0, 0
    for i, (input, target, src_imgs, _) in enumerate(data_loader):
        count += len(target)
        output = model(input.cuda())
        acc, find_idxs = find_idx(output, center, target)
        acc_sum += acc
        logging.info(f'{i}-{acc}')
        find_files = find_src_img(os.path.join(test_path, GlobalConfig.TRAIN),
                                  idx2label(idx_map, find_idxs),
                                  _type=data_type)
        merge_pic_trace(src_imgs, find_files, f'test/images/{i}_{data_type}_trace.jpg')
    logging.info(f'csq acc:{acc_sum / count}')


def idx2label(map: {}, find_idxs):
    labels = []
    for idx in find_idxs:
        labels.append(map[idx])
    return labels


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


def find_src_img(path, label: [], _type=0):
    files = []
    for l_ in label:
        path_f = os.path.join(path, l_, 'src') if _type == 0 else os.path.join(path, 'src', l_)
        file = os.listdir(path_f)[0]
        file = os.path.join(path_f, file)
        files.append(file)

    return files


def find_idx(output, hash_center, target):
    output = output.data.cpu().float()
    real_count, find_labels, f_l = 0, [], None
    for j in range(len(target)):
        l_ = target[j]
        f_l = l_
        out = output[j]
        min_loss = GlobalConfig.HASH_BIT  # hash bit
        for i in range(len(hash_center)):
            bit = hash_center[i]
            loss = criterion(0.5 * (out + 1), 0.5 * (bit + 1))
            if loss < min_loss:
                min_loss = loss
                f_l = i
        if l_ == f_l:
            real_count += 1
        else:
            logging.info(f'error:{l_}, {f_l}, {min_loss}')
        find_labels.append(f_l)
    return real_count , find_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Hash Test')
    parser.add_argument('--dataset', default='VRF', choices=['VRF', 'Inpainting'],
                        help="path to dataset")
    parser.add_argument('--gpus', type=str, default="0",
                        help="define gpu id")
    # hash
    parser.add_argument('--hash_bit', type=int, default=GlobalConfig.HASH_BIT,
                        help="define the length of hash bits.")
    parser.add_argument('--batch_size', type=int, default=GlobalConfig.BATCH_SIZE)
    # initialization with priority (the next step will overwrite the previous step)
    # - step 1: random initialize
    # - step 2: load the 2D pretrained model if `pretrained_2d' is True
    # - step 3: load the 3D pretrained model if `pretrained_3d' is defined
    # - step 4: resume if `resume_epoch' >= 0
    parser.add_argument('--pretrained_2d', type=bool, default=False,
                        help="load default 2D pretrained model.")
    parser.add_argument('--pretrained_3d', type=str,
                        default='./exps/models/PyTorch-MFNet-master_ep-0013.pth',
                        help="load default 3D pretrained model.")
    parser.add_argument('--resume-epoch', type=int, default=-1,
                        help="resume train")
    parser.add_argument('--network', type=str, default='MFNet_3D',
                        choices=['MFNet_3D'],
                        help="chose the base network")

    # distributed training
    parser.add_argument('--backend', default='nccl', type=str, choices=['gloo', 'nccl'],
                        help='Name of the backend to use')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://192.168.0.11:23456', type=str,
                        help='url used to set up distributed training')
    # calculate MAP
    parser.add_argument('--R', default=100, type=int)
    parser.add_argument('--T', default=0, type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # before using torch
    assert torch.cuda.is_available(), "CUDA is not available"
    Hash_center = None
    data_type = 0
    path = '../../vrf'
    if args.dataset == 'VRF':
        Hash_center = torch.load('dataset/64_vrf_133_class.pkl')
    if args.dataset == 'Inpainting':
        data_type = 1
        path = '../../inpainting'
        Hash_center = torch.load('dataset/64_inpainting_50_class.pkl')
    dataset_cfg = dataset.get_config(name=args.dataset)
    dataset_name = args.dataset

    batch_size = args.batch_size
    distributed = False
    resume_epoch = -1

    net, input_conf = get_symbol(name=args.network,
                                 pretrained=args.pretrained_2d if args.resume_epoch < 0 else None,
                                 print_net=False,
                                 hash_bit=args.hash_bit,
                                 **dataset_cfg)
    net.eval()
    net = net.cuda()
    checkpoint = torch.load(args.pretrained_3d)
    net.load_state_dict(checkpoint['state_dict'])

    train_iter, eval_iter = iterator_factory.creat(name=dataset_name, data_type=0 if dataset_name == 'VRF' else 1,
                                                   batch_size=batch_size)
    predict_similar(net, eval_iter, Hash_center, test_path=path, data_type=data_type)
