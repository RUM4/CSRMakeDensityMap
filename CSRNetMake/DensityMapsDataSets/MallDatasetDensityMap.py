# -*- coding:utf-8 -*-
"""
@author:lpf
@file: MallDatasetDensityMap.py
@time: 2020/03/08
"""
import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
from image import *
#https://github.com/davideverona/deep-crowd-counting_crowdnet
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

mall_dataset = os.path.join(root,'mall_dataset/','frames')
path_sets = [mall_dataset]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
mat_paths = os.path.join(root,'mall_dataset/','mall_gt.mat')
mat = io.loadmat(mat_paths)
gt = mat["frame"][0]
print(len(gt),len(img_paths),gt[0][0][0][0][0][0],gt[1][0][0][0][0][0])

for img_path in img_paths:
    #读取程numpy数据
    img= plt.imread(img_path)
    #构建一个和img相同维度的numpy
    k = np.zeros((img.shape[0],img.shape[1]))
    #读取mat文件内容
    for i in range(0,len(gt)):
        print("gt,img", gt[i][0][0][0][0][1], img.shape[0])
        if int(gt[i][0][0][0][0][1]) < img.shape[0] and int(gt[i][0][0][0][0][0]) < img.shape[1]:
            k[int(gt[i][0][0][0][0][1]),int(gt[i][0][0][0][0][0])]=1
    k = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.jpg','.h5'), 'w') as hf:
            hf['density'] = k