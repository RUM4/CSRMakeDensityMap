# -*- coding:utf-8 -*-
"""
@author:lpf
@file: OneImageTest.py
@time: 2020/03/04
"""
from matplotlib import cm as c

img = transform(Image.open('part_A/test_data/images/IMG_100.jpg').convert('RGB')).cuda()

output = model(img.unsqueeze(0))

print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))

temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))

plt.imshow(temp,cmap = c.jet)

plt.show()

temp = h5py.File('part_A/test_data/ground-truth/IMG_100.h5', 'r')

temp_1 = np.asarray(temp['density'])

plt.imshow(temp_1,cmap = c.jet)

print("Original Count : ",int(np.sum(temp_1)) + 1)

plt.show()

print("Original Image")

plt.imshow(plt.imread('part_A/test_data/images/IMG_100.jpg'))

plt.show()
