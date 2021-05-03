# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：hist.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：01/05/2021 14:47 
"""
from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread('./meanshift.jpg')
colors = ('b', 'g', 'r')
chans = cv2.split(img)
# plt.imshow(img_gray, cmap=plt.cm.gray)


plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
# plt.plot(hist)
# plt.xlim([0, 256])
# plt.show()
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 255])
    plt.plot(hist, c=color)
    plt.xlim([0, 256])
    print(hist)

plt.show()

