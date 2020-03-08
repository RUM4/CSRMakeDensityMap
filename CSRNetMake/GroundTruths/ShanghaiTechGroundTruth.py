# -*- coding:utf-8 -*-
"""
@author:lpf
@file: ShanghaiTechGroundTruth.py
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
part_A_train = os.path.join(root,'ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data','images')
part_A_test = os.path.join(root,'ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data','images')
path_sets = [part_A_train,part_A_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(img_paths[0])
plt.imshow(Image.open(img_paths[0]))

gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground_truth'),'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)
print(np.sum(groundtruth))
plt.show()
