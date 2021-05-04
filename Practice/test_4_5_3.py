# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：test_4_5_3.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：04/05/2021 11:24 
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
b_less, g_less, r_less = cv2.split(img_less)
b_less = cv2.equalizeHist(b_less)
g_less = cv2.equalizeHist(g_less)
r_less = cv2.equalizeHist(r_less)
img_less = cv2.merge([r_less, g_less, b_less])

img_over = cv2.imread(conf.items("Path")[3][1])
b_o, g_o, r_o = cv2.split(img_over)
b_o, g_o, r_o = cv2.equalizeHist(b_o), cv2.equalizeHist(g_o), cv2.equalizeHist(
    r_o)
img_over = cv2.merge([r_o, g_o, b_o])

plt.subplot(1, 2, 1)
plt.imshow(img_less)
plt.title("Less Exposure")

plt.subplot(1, 2, 2)
plt.imshow(img_over)
plt.title("Over Exposure")

plt.show()