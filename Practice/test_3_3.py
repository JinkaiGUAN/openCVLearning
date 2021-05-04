# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：test_3_3.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：03/05/2021 12:31 
"""
import cv2
import matplotlib.pyplot as plt
import configparser
import os

# Gets the configuration file path
dir_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(dir_path, "config.ini")

# initiate the configuration path
conf = configparser.ConfigParser()
conf.read(filenames=config_path, encoding='utf-8')

persimmon_img = cv2.imread(conf.items("Path")[0][1], cv2.IMREAD_UNCHANGED)

# ------------------------------------------------------------------------- #
# Gray histogram for persimmon
# persimmon_img_gray = cv2.cvtColor(persimmon_img, cv2.COLOR_BGR2GRAY)
# gray_persimmon_hist = cv2.calcHist(persimmon_img_gray, channels=[0], mask=None,
#                                    histSize=[256], ranges=[0, 255])
#
# plt.plot(gray_persimmon_hist)
# plt.show()

# ------------------------------------------------------------------------- #
# Colorful Histogram for persimmon
# channels = cv2.split(persimmon_img)
# colors = ['b', 'g', 'r']
#
#
# for i, channel in enumerate(channels):
#     hist_graph = cv2.calcHist(images=[channel], channels=[0], mask=None,
#                               histSize=[256], ranges=[0, 255])
#     plt.plot(hist_graph, c=colors[i])
#
# plt.xlabel("Bin")
# plt.ylabel("Pixel Num")
# plt.title("Persimmon Histogram for Three Channels")
# plt.show()

# -------------------------------------------------------------------------- #
# HSV demo
# persimmon_img_hsv = cv2.cvtColor(persimmon_img, cv2.COLOR_BGR2HSV)
# plt.imshow(persimmon_img_hsv)
# plt.show()

# -------------------------------------------------------------------------- #
# CIELab demo
# persimmon_img_lab = cv2.cvtColor(persimmon_img, cv2.COLOR_BGR2Lab)
# persimmon_img_LAB = cv2.cvtColor(persimmon_img, cv2.COLOR_BGR2LAB)
# plt.imshow(persimmon_img_LAB)
# plt.show()

# -------------------------------------------------------------------------- #
# Foreground and Background demo
persimmon_img_gray = cv2.cvtColor(persimmon_img, cv2.COLOR_BGR2GRAY)
ret, persimmon_img_foreground = cv2.threshold(persimmon_img_gray, thresh=100,
                                              maxval=255,
                                              type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(persimmon_img_foreground)
plt.show()
# persimmon_img_thresh = cv2.threshold(persimmon_img, 100, 255, cv2.THRESH_BINARY)[1]
