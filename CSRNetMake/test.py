# -*- coding:utf-8 -*-
"""
@author:lpf
@file: test.py
@time: 2020/03/04
"""
#importing libraries

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

import torchvision.transforms.functional as F

from matplotlib import cm as CM

from image import *

from model import CSRNet

import torch


from torchvision import datasets, transforms

transform=transforms.Compose([

transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],

std=[0.229, 0.224, 0.225]),

])

#defining the location of dataset

root = '/home/lpf/PycharmProjects/MakeDensityMap/ShanghaiTech_Crowd_Counting_Dataset/'

part_A_train = os.path.join(root,'part_A/train_data','images')

part_A_test = os.path.join(root,'part_A/test_data','images')

part_B_train = os.path.join(root,'part_B/train_data','images')

part_B_test = os.path.join(root,'part_B/test_data','images')

path_sets = [part_A_test]

#defining the image path

img_paths = []

for path in path_sets:

for img_path in glob.glob(os.path.join(path, '*.jpg')):

img_paths.append(img_path)

model = CSRNet()

#defining the model

model = model.cuda()

#loading the trained weights

checkpoint = torch.load('part_A/0model_best.pth.tar')

model.load_state_dict(checkpoint['state_dict'])

#检查测试图像上的MAE（平均绝对误差），评估我们的模型：
mae = 0

for i in tqdm(range(len(img_paths))):

img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()

gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground-truth'),'r')

groundtruth = np.asarray(gt_file['density'])

output = model(img.unsqueeze(0))

mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))

print (mae/len(img_paths))