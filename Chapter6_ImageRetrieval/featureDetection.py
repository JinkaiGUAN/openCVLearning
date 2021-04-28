# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：featureDetection.py.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：25/04/2021 19:40 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

model_name = 'c'
img = cv2.imread('../../images/cheese.png')
img = cv2.pyrDown(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, blockSize=2, ksize=23, k=0.04)
img[dst>0.01 * dst.max()] = [0, 0, 255]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
