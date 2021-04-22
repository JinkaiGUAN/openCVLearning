# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：gradCutAlgorithm.py.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：18/04/2021 12:06 
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("../../images/Dress.JPG")
mask = np.zeros(img.shape[:2], np.uint8)

"""the size of the background and foreground models is the specific number,
should be changed."""
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

rect = (100, 100, 900, 500)
cv2.grabCut(img, mask=mask, rect=rect, bgdModel=bgd_model, fgdModel=fgd_model,
            iterCount=10, mode=cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]

plt.subplot(121)
plt.imshow(img)
plt.title("grabcut")
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(
    cv2.cvtColor(cv2.imread("../../images/Dress.JPG"), cv2.COLOR_BGR2RGB))
plt.title("original")
plt.xticks([]), plt.yticks([])
plt.show()
