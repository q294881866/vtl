import os
from glob import glob

import cv2 as cv

mask_path = './youtube-vos/train/Annotations'
save_path = './youtube-vos/train/mask'

if not os.path.isdir(save_path):
    os.makedirs(save_path)
t = 0
files = os.listdir(mask_path)
for file in files:
    mask_file = os.path.join(mask_path, file)
    save_file = os.path.join(save_path, file)

    if not os.path.isdir(save_file):
        os.makedirs(save_file)
    flag = 0
    # if t > 2:
    #     break
    for mask in glob(mask_file + "/*.png"):
        read_mask = cv.imread(mask)
        x = read_mask.shape[0]
        y = read_mask.shape[1]

        for i in range(x):
            for j in range(y):
                if read_mask[i, j, 0] > 0 or read_mask[i, j, 1] > 0 or read_mask[i, j, 2] > 0:
                    read_mask[i, j, 0] = 255
                    read_mask[i, j, 1] = 255
                    read_mask[i, j, 2] = 255

        cv.imwrite(save_file + "/" + str(flag).zfill(5) + ".png", read_mask)
        flag = flag + 5
    t = t + 1
    print("Finished processing input {k}.".format(k=mask_file))
