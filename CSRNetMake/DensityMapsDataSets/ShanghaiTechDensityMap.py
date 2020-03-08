# -*- coding:utf-8 -*-
"""
@author:lpf
@file: ShanghaiTechDensityMap.py
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
part_A_train = os.path.join(root,'ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data','images')
part_A_test = os.path.join(root,'ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data','images')
part_B_train = os.path.join(root,'ShanghaiTech_Crowd_Counting_Dataset/part_B_final/train_data','images')
part_B_test = os.path.join(root,'ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data','images')
path_sets = [part_A_train,part_A_test]

#获取路径下所有图片的路径
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
# 一,ShanghaiTech_DataSet
# 1,partA部分
for img_path in img_paths:
    #产生图像对应mat路径
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    print(mat,type(mat)) #dict类型
    #读取程numpy数据
    img= plt.imread(img_path)
    #构建一个和img相同维度的numpy
    k = np.zeros((img.shape[0],img.shape[1]))
    #读取mat文件内容
    gt = mat["image_info"][0,0][0,0][0]
    print('gt',gt) #坐标系内容
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k

# 2,partB部分
# path_sets = [part_B_train,part_B_test]
# img_paths = []
# for path in path_sets:
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):
#         img_paths.append(img_path)
# for img_path in img_paths:
#     print (img_path)
#     mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
#     img= plt.imread(img_path)
#     k = np.zeros((img.shape[0],img.shape[1]))
#     gt = mat["image_info"][0,0][0,0][0]
#     for i in range(0,len(gt)):
#         if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
#             k[int(gt[i][1]),int(gt[i][0])]=1
#     k = gaussian_filter(k,15)
#     with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
#             hf['density'] = k
