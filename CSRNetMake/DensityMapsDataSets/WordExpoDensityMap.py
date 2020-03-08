# -*- coding:utf-8 -*-
"""
@author:lpf
@file: WordExpoDensityMap.py
@time: 2020/03/08
"""
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from tqdm import tqdm
#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    print(gt_count)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048

    # build kdtree 寻找最临近点
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

root = '/home/lpf/PycharmProjects/MakeDensityMap/'
#二,WorldExpo_DataSet
WorldExpoTest104207 = os.path.join(root,'WorldExpo/test/test_frame/','104207')
# WorldExpoTest200608 = os.path.join(root,'WorldExpo/test/test_frame/','200608')
# WorldExpoTest200702 = os.path.join(root,'WorldExpo/test/test_frame/','200702')
# WorldExpoTest202201 = os.path.join(root,'WorldExpo/test/test_frame/','202201')
# WorldExpoTest500717 = os.path.join(root,'WorldExpo/test/test_frame/','500717')

path_sets = [WorldExpoTest104207]
# path_sets = [WorldExpoTest200608]
# path_sets = [WorldExpoTest200702]
# path_sets = [WorldExpoTest202201]
# path_sets = [WorldExpoTest500717]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
for img_path in img_paths:
    mat_path = img_path.replace('.jpg', '.mat').replace('test_frame', 'test_label')
    mat = io.loadmat(mat_path)
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["point_position"]

    for i in range(0,len(gt)):
        print("gt,img",gt[i][1],img.shape[0])
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter(k,3)
    with h5py.File(img_path.replace('.jpg','.h5'), 'w') as hf:
            hf['density'] = k