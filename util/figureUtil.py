import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import Tensor

from layer.helper import to_image


def analyze_hash_dist(name, itr_times, hash_dists):
    plt.title('Hash Analysis')
    plt.plot(itr_times, hash_dists, color='black', label='hash distance')
    plt.legend()

    plt.xlabel('iteration times')
    plt.ylabel('value')
    plt.draw()
    plt.savefig(name)  # 保存图象
    plt.close()


def analyze_loss(name, itr_times, h_losses, d_losses, accuracies):
    plt.title('Loss Analysis')
    plt.plot(itr_times, h_losses, color='skyblue', label='h_losses')
    plt.plot(itr_times, d_losses, color='red', label='d_losses')
    plt.plot(itr_times, accuracies, color='yellow', label='accuracies')
    plt.legend()

    plt.xlabel('iteration times')
    plt.ylabel('loss rate')
    plt.draw()
    plt.savefig(name)
    plt.close()


def analyze_acc():
    values = ['64bits', '128bits', '256bits', '512bits', '1024bits']
    acc64 = [0.76424, 0.76518, 0.76500, 0.76481, 0.76537, 0.71296]
    acc128 = [0.82289, 0.82308, 0.82289, 0.82271, 0.82195, 0.71088]
    acc256 = [0.93869, 0.93907, 0.93756, 0.93737, 0.93945, 0.87398]
    acc512 = [0.98269, 0.98297, 0.98127, 0.98165, 0.98278, 0.97483]
    acc1024 = [0.93813, 0.93794, 0.93718, 0.93718, 0.93775, 0.94078]
    acc = [np.array(acc64).mean(),
           np.array(acc128).mean(),
           np.array(acc256).mean(),
           np.array(acc512).mean(),
           np.array(acc1024).mean(),
           ]
    in_acc64 = [0.76651, 0.76613, 0.76689, 0.76632, 0.76708, 0.71485]
    in_acc128 = [0.90162, 0.90015, 0.90308, 0.90308, 0.89868, 0.87665]
    in_acc256 = [0.90896, 0.90896, 0.90455, 0.90749, 0.91043, 0.88546]
    in_acc512 = [0.89280, 0.89280, 0.89574, 0.89574, 0.89427, 0.90015]
    in_acc1024 = [0.90455, 0.90308, 0.90455, 0.90455, 0.90455, 0.87225]
    in_acc = [np.array(in_acc64).mean(),
              np.array(in_acc128).mean(),
              np.array(in_acc256).mean(),
              np.array(in_acc512).mean(),
              np.array(in_acc1024).mean(),
              ]
    plt.plot(values, acc, color='skyblue', label='DFTL')
    plt.plot(values, in_acc, color='green', label='DAVIS2016-TL')
    plt.ylabel('Top-1 Accuracy')
    plt.legend()

    plt.draw()
    plt.savefig('acc.png')  # 保存图象
    plt.close()


def merge_pic(g_tensor: Tensor, mask: Tensor, name):
    b, t, h, w = g_tensor.shape
    g_tensor = g_tensor.detach().cpu()
    img = to_image(g_tensor[0][0])
    images = Image.new(img.mode, (w * t, h * b * 2))
    for j in range(b * 2):
        for i in range(t):
            img = to_image(g_tensor[j // 2][i] if j % 2 == 0 else mask[j // 2][i])
            images.paste(img, box=(i * w, j * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def samples():
    r = 50
    xs, ys = [], []
    for i in range(1000):
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if x ** 2 + y ** 2 < 50 ** 2:
            xs.append(x)
            ys.append(y)
    plt.scatter(xs, ys, s=2, facecolors='none', edgecolors='r')
    plt.legend()

    plt.draw()
    plt.savefig('acc.png')  # 保存图象
    plt.show()
    plt.close()


if __name__ == '__main__':
    samples()
