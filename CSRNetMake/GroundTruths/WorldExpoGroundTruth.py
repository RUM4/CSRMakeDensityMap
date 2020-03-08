# -*- coding:utf-8 -*-
"""
@author:lpf
@file: WorldExpoGroundTruth.py
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
WorldExpoTest104207 = os.path.join(root,'WorldExpo/test/test_frame/','104207')
# WorldExpoTest200608 = os.path.join(root,'WorldExpo/test/test_frame/','200608')
# WorldExpoTest200702 = os.path.join(root,'WorldExpo/test/test_frame/','200702')
# WorldExpoTest202201 = os.path.join(root,'WorldExpo/test/test_frame/','202201')
# WorldExpoTest500717 = os.path.join(root,'WorldExpo/test/test_frame/','500717')

path_sets = [WorldExpoTest104207]
print(path_sets)
#获取路径下所有图片的路径
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(img_paths[1])
plt.imshow(Image.open(img_paths[1]))

gt_file = h5py.File(img_paths[1].replace('.jpg','.h5'),'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)
print(np.sum(groundtruth))
plt.show()
