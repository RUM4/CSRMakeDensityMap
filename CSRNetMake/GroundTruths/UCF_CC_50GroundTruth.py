# -*- coding:utf-8 -*-
"""
@author:lpf
@file: UCF_CC_50GroundTruth.py
@time: 2020/03/08
"""
import h5py
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from image import *

root = '/home/lpf/PycharmProjects/MakeDensityMap/'
UCF_CC_50_h5_paths = [os.path.join(root,'UCF_CC_50')]
h5_paths =[]
for h5_path in UCF_CC_50_h5_paths:
    for h5_path in glob.glob(os.path.join(h5_path, '*.h5')):
        h5_paths.append(h5_path)
print(h5_paths[0])
gt_file = h5py.File(h5_paths[0],'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)
print(np.sum(groundtruth))
plt.show()