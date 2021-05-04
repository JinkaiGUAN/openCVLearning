# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：test_4_5_2.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：04/05/2021 10:14 
"""
import os
import cv2
import configparser
import matplotlib.pyplot as plt
import numpy as np

__doc__ = "Simple test for Gamma transformation."

# Read the image
dir_name = os.path.dirname(os.path.abspath(__file__))
conf_path = os.path.join(dir_name, "config.ini")
conf = configparser.ConfigParser()
conf.read(conf_path, encoding='utf-8')

img_less = cv2.imread(conf.items("Path")[1][1])
img_less_ = cv2.cvtColor(img_less, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(10, 9))
plt.subplot(4, 3, 1)
plt.imshow(img_less_)
plt.title("Less ori")

gammas = [1.0, 2.0, 3.0, 4.0, 5.0]

img_less_nor = img_less/float(np.max(img_less)) # normalisation
for i, gamma in enumerate(gammas):
    img_less_gamma = np.power(img_less_nor, 1 / gamma)
    b, g, r = cv2.split(img_less_gamma)
    img_less_gamma = cv2.merge([r, g, b])
    plt.subplot(4, 3, i+2)
    plt.imshow(img_less_gamma)
    plt.title("gamma: 1/{:.1f}".format(gamma))

img_over = cv2.imread(conf.items("Path")[3][1])
img_over_nor = img_over / float(np.max(img_over))
for i, gamma in enumerate(gammas):
    img_over_gamma = np.power(img_over_nor, gamma)
    b, g, r = cv2.split(img_over_gamma)
    img_over_gamma = cv2.merge([r, g, b])
    plt.subplot(4, 3, 7+i)
    plt.imshow(img_over_gamma)
    plt.title("gamma: {:.1f}".format(gamma))

img_normal = cv2.imread(conf.items("Path")[2][1])
img_normal = cv2.cvtColor(img_normal, cv2.COLOR_BGR2RGB)
plt.subplot(4, 3, 12)
plt.imshow(img_normal)
plt.title("Original")

plt.tight_layout()
plt.subplots_adjust(wspace =0.0, hspace =0.4)#调整子图间距
plt.show()

