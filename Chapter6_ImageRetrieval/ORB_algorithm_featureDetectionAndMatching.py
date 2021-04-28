# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：ORB_algorithm_featureDetectionAndMatching.py.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：25/04/2021 20:13 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_small = cv2.imread('small.jpg', flags=cv2.IMREAD_GRAYSCALE)
img_big = cv2.imread('big.jpg', flags=cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
keypoint_small, descriptors_small = orb.detectAndCompute(img_small, None)
keypoint_big, descriptors_big = orb.detectAndCompute(img_big, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors_small, descriptors_big)
matches = sorted(matches, key=lambda x: x.distance)
img_match = cv2.drawMatches(img1=img_small, keypoints1=keypoint_small,
                            img2=img_big, keypoints2=keypoint_big,
                            matches1to2=matches[:40], outImg=img_big, flags=2)
plt.imshow(img_match)
plt.show()
