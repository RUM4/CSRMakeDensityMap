# -*- coding:utf-8 -*-
"""
@author:lpf
@file: MallDataSetGroundTruth.py
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
mall_dataset = os.path.join(root,'mall_dataset/','frames')
path_sets = [mall_dataset]
print(path_sets)
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(img_paths[3])
plt.imshow(Image.open(img_paths[3]))

gt_file = h5py.File(img_paths[3].replace('.jpg','.h5'),'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)
print(np.sum(groundtruth))
plt.show()