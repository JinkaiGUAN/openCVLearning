# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：test_4_5_4.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：04/05/2021 11:42 
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

img = cv2.imread(conf.items("Path")[0][1])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(2, 3, 1)
plt.imshow(img_gray)
plt.title("Original")

# ---------------------- sobel --------------------------- #
sobelx = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # the gradient of x
sobely = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # the gradient of y

sobelx = np.uint8(np.absolute(sobelx))
sobley = np.uint8(np.absolute(sobely))
sobelxy = cv2.bitwise_or(sobelx, sobley)

sobels = [img_gray, sobelx, sobely, sobelxy]
sobel_titles = ['img', 'soblex', 'sobely', 'sobelxy']
for i, sobel in enumerate(sobels):
    plt.subplot(2, 4, 1+i)
    plt.imshow(sobel, cmap="gray")
    plt.title(sobel_titles[i])

# ------------------------ Laplacian -------------------- #
laplacian = cv2.Laplacian(img_gray, ddepth=cv2.CV_64F, ksize=5)
laplacian = np.uint8(np.absolute(laplacian))
plt.subplot(2, 4, 5)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian")

# -------------------------- Canny --------------------- #
canny = cv2.Canny(img_gray, threshold1=100, threshold2=150)
plt.subplot(2, 4, 6)
plt.imshow(canny, cmap="gray")
plt.title("Canny")

# ------------------------ prewitt ---------------------- #
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewittx = cv2.filter2D(img_gray, ddepth=-1, kernel=kernelx)
prewitty = cv2.filter2D(img_gray, ddepth=-1, kernel=kernely)
prewitt = cv2.bitwise_or(prewittx, prewitty)

plt.subplot(2, 4, 7)
plt.imshow(prewitt, cmap="gray")
plt.title("Prewitt")

plt.show()







