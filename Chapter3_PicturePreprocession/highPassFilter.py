# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：highPassFilter.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：11/04/2021 12:29 
"""
import cv2
import numpy as np
from scipy import ndimage

kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, 2, 4, 2, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, -1, -1, -1, -1]])
img = cv2.imread("../../images/Labrad.jpg", 0)

k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)

blurred = cv2.GaussianBlur(img, (11, 11), 0)
g_hpf = img - blurred
cv2.imshow("Original", img)
cv2.imshow("3x3", k3)
cv2.imshow("5x5", k5)
cv2.imshow("g_hpf", g_hpf)
cv2.waitKey()
cv2.destroyAllWindows()



















