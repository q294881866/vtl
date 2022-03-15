import os
import random

import cv2

source_path = 'E:/manual/SOURCE'
jpg_path = 'E:/manual/JPEGImages'
annotations_path = 'E:/manual/Annotations'

fake_path = './fake'
mask_path = './mask'

size = [512, 960]

if not os.path.isdir(fake_path):
    os.makedirs(fake_path)
if not os.path.isdir(mask_path):
    os.makedirs(mask_path)


def make_splicing():
    for item in os.listdir(source_path):
        item_path = os.path.join(source_path, item)
        files = os.listdir(item_path)
        count = 0
        for mark in os.listdir(jpg_path):
            mark_dir = os.path.join(jpg_path, mark)
            mask_dir = os.path.join(annotations_path, mark)
            mark_files = os.listdir(mark_dir)
            scale = random.randint(5, 10) / 10
            pos = (random.randint(200, 700), random.randint(150, 350))
            if len(files) <= len(mark_files) and count < 4:
                count += 1
                for i in range(len(files)):
                    file = files[i]
                    target_file = os.path.join(item_path, file)
                    save_dir = os.path.join(fake_path, item, mark)
                    save_mask_dir = os.path.join(mask_path, item, mark)

                    mark_file = mark_files[i]
                    mask_file = os.path.join(mask_dir, mark_file)
                    source_file = os.path.join(mark_dir, mark_file)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    if not os.path.exists(save_mask_dir):
                        os.makedirs(save_mask_dir)

                    sync_mask(source_file, mask_file, target_file, os.path.join(save_dir, file), os.path.join(save_mask_dir, file), scale, pos)
                    print("Finished processing input {k}.".format(k=source_file))


def sync_mask(source, mask, target, save_file, save_mask_file, scale, pos):
    read_source = cv2.imread(source)
    read_mask = cv2.imread(mask)
    read_target = cv2.imread(target)

    read_source = cv2.resize(read_source, (960, 512))
    read_mask = cv2.resize(read_mask, (960, 512))
    read_target = cv2.resize(read_target, (960, 512))

    # 图像缩放
    scale_img = cv2.resize(read_source, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scale_mask = cv2.resize(read_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # 图像裁剪
    x, y = pos
    if x - 0.5 * scale * size[1] > 0:
        x0 = 0
    else:
        x0 = scale * size[1] * 0.5 - x
    if x + 0.5 * scale * size[1] < size[1]:
        x1 = scale * size[1]
    else:
        x1 = 0.5 * scale * size[1] + size[1] - x

    if y - 0.5 * scale * size[0] > 0:
        y0 = 0
    else:
        y0 = scale * size[0] * 0.5 - y
    if y + 0.5 * scale * size[0] < size[0]:
        y1 = scale * size[0]
    else:
        y1 = 0.5 * scale * size[0] + size[0] - y
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)

    cut_img = scale_img[y0:y1, x0:x1]
    cut_mask = scale_mask[y0:y1, x0:x1]

    # 图像补全
    if y0 == 0:
        top = y - scale * size[0] * 0.5
    else:
        top = 0
        bottom = size[0] - y - scale * size[0] * 0.5
    if y1 == scale * size[0]:
        bottom = size[0] - y - scale * size[0] * 0.5
    else:
        bottom = 0
        top = y - scale * size[0] * 0.5

    if x0 == 0:
        left = x - scale * size[1] * 0.5
    else:
        left = 0
        right = size[1] - x - scale * size[1] * 0.5
    if x1 == scale * size[1]:
        right = size[1] - x - scale * size[1] * 0.5
    else:
        right = 0
        left = x - scale * size[1] * 0.5

    top, bottom, left, right = int(top), int(bottom), int(left), int(right)

    filled_img = cv2.copyMakeBorder(cut_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    filled_mask = cv2.copyMakeBorder(cut_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 翻转
    if scale * 10 // 2 == 0:
        filled_img = cv2.flip(filled_img, 1)
        filled_mask = cv2.flip(filled_mask, 1)

    filled_mask = filled_mask // 255
    filled_img = cv2.resize(filled_img, (960, 512))
    filled_mask = cv2.resize(filled_mask, (960, 512))
    out_1 = filled_img * filled_mask
    out_2 = read_target * (1 - filled_mask)
    out = out_1 + out_2

    cv2.imwrite(save_file, out)
    cv2.imwrite(save_mask_file, filled_mask * 255)


if __name__ == '__main__':
    make_splicing()
