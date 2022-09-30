# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/9/9 10:55
# @Author : liumin
# @File : generate_sine_wave.py

import numpy as np
import torch

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')
torch.save(data, open('traindata.pt', 'wb'))