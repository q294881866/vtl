# random generation
import torch
import random
import numpy as np
import csv
from scipy.special import comb, perm  # calculate combination
from itertools import combinations
import GlobalConfig
import layer.helper

hash_targets = []
a = []  # for sampling the 0.5*hash_bit
b = []  # for calculate the combinations of 51 num_class
num_class = 133
hash_bit = 64

for i in range(0, hash_bit):
    a.append(i)

for i in range(0, num_class):
    b.append(i)

for j in range(10000):
    hash_targets = torch.zeros([num_class, hash_bit])
    for i in range(num_class):
        ones = torch.ones(hash_bit)
        sa = random.sample(a, round(hash_bit / 2))
        ones[sa] = -1
        hash_targets[i] = ones
    com_num = int(comb(num_class, 2))
    c = np.zeros(com_num)
    for i in range(com_num):
        i_1 = list(combinations(b, 2))[i][0]
        i_2 = list(combinations(b, 2))[i][1]
        TF = torch.sum(hash_targets[i_1] != hash_targets[i_2])
        c[i] = TF
    print(min(c))
    print(max(c))
    print(np.mean(c))

    # guarantee the hash center are far away from each other in Hamming space, 20 can be set as 18 for fast convergence
    if min(c) >= 20 and np.mean(c) >= 32:
        print(min(c))
        print("stop! we find suitable hash centers")
        break



# Save the hash targets as training targets
file_name = str(hash_bit) + '_vrf' + '_' + str(num_class) + '_class.pkl'
file_dir = './' + file_name
f = open(file_dir, "wb")

if __name__ == '__main__':
    print(hash_targets.shape)
    torch.save(hash_targets, f)