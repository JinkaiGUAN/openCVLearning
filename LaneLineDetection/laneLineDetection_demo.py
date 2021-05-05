# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：laneLineDetection_demo.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：04/05/2021 14:54 
"""
import os
import cv2
import configparser
import numpy as np
import matplotlib.pyplot as plt

dir_name = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(dir_name, "config.ini")
conf = configparser.ConfigParser()
conf.read(config_path, encoding='utf-8')

# Canny -> edge detection
img = cv2.imread(conf.items("Path")[0][1])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gauss = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=0, sigmaY=0)
canny_edge = cv2.Canny(img_gauss, threshold1=int(conf.items("Canny")[0][1]),
                       threshold2=int(conf.items("Canny")[1][1]),
                       apertureSize=3, L2gradient=True)
# canny = cv2.Canny(img_gauss, threshold1=int(conf.items("Canny")[0][1]), threshold2=150)
# plt.imshow(canny_edge, cmap="gray")

# Straight line detection
lines = cv2.HoughLines(img_gray, rho=1, theta=np.pi / 180, threshold=200)

for rho, theta in lines[0]:
    a = np.cos(theta)  # x 偏移
    b = np.sin(theta)  # y 偏移
    x0 = rho * a
    y0 = rho * b

    # 右无穷点
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))

    ##左无穷点
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.circle(img, (int(x0), int(y0)), radius=50, color=(255, 255, 0),
               thickness=2, lineType=8, shift=0)  ##画出切点
    cv2.line(img, (0, 0), (int(x0), int(y0)), (255, 0, 0), 2)  ##画出半径
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  ##画出直线

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
