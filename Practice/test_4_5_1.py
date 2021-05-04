# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：test_4_5_1.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：04/05/2021 08:27
"""
import os
import cv2
import configparser
import matplotlib.pyplot as plt

__doc__ = "Simple test for reducing the noise impact."

# Read the image
dir_name = os.path.dirname(os.path.abspath(__file__))
conf_path = os.path.join(dir_name, "config.ini")
conf = configparser.ConfigParser()
conf.read(conf_path, encoding='utf-8')

img = cv2.imread(conf.items("Path")[0][1], cv2.IMREAD_UNCHANGED)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------------- Gaussian ---------------------------------- #
img_gaussian = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
img_gaussian = cv2.cvtColor(img_gaussian, cv2.COLOR_BGR2RGB)
plt.subplot(341)
plt.imshow(img_gaussian)
plt.title("Gaussian Blur")

# ---------------------- Median Value ------------------------------ #
img_median = cv2.medianBlur(img, ksize=5)
img_median = cv2.cvtColor(img_median, cv2.COLOR_BGR2RGB)
plt.subplot(343)
plt.imshow(img_median)
plt.title("Median Blur")

# --------------------- Bilateral --------------------------------- #
img_bilateral = cv2.bilateralFilter(img, d=10, sigmaColor=40, sigmaSpace=40)
img_bilateral = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2RGB)
plt.subplot(345)
plt.imshow(img_bilateral)
plt.title("Bilateral Filter")

# ------------------------ Non local mean filter ------------------ #
img_nlm = cv2.fastNlMeansDenoisingColored(img, h=10, templateWindowSize=10,
                                   searchWindowSize=7)
img_nlm = cv2.cvtColor(img_nlm, cv2.COLOR_BGR2RGB)
plt.subplot(3, 4, 7)
plt.imshow(img_nlm)
plt.title("Non local mean")

plt.show()
