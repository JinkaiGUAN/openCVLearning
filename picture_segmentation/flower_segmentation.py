# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：flower_segmentation.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：05/05/2021 19:27 
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import configparser

dir_path = os.path.dirname(os.path.abspath(__file__))
conf_path = os.path.join(dir_path, "config.ini")
conf = configparser.ConfigParser()
conf.read(conf_path, encoding='utf-8')


class FlowerSegmentation(object):
    """
    Segmentation with several kinds of color space
    """

    def __init__(self, color_space="gray"):
        self.color_space = color_space

    def __del__(self):
        print("The flower segmentation instance is deleted!")

    def ostuSegmentation(self, img):
        """
        Using ostu method to segment the image with various kinds of color
         space.

        :param img: rgb image.
        :return:
        """
        if self.color_space == "gray":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0,
                                        sigmaY=0)
            self.ret, self.thresh_img = cv2.threshold(img_blur, thresh=0,
                                                      maxval=255,
                                                      type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.color_space == "hsv":
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv = img_hsv[:, :, 2]
            img_blur = cv2.GaussianBlur(img_hsv, ksize=(5, 5), sigmaX=0,
                                        sigmaY=0)
            self.ret, self.thresh_img = cv2.threshold(img_blur, thresh=0,
                                                      maxval=255,
                                                      type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.color_space == "lab":
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img_lab = img_lab[:, :, 1]
            img_blur = cv2.GaussianBlur(img_lab, ksize=(5, 5), sigmaX=0,
                                        sigmaY=0)
            self.ret, self.thresh_img = cv2.threshold(img_blur, thresh=0,
                                                      maxval=255,
                                                      type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def grabCut(self, image):
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        rect = (100, 100, 500, 400)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[self.thresh_img == 0] = cv2.GC_PR_BGD
        mask[self.thresh_img == 255] = cv2.GC_PR_FGD
        mask, bgd, fgd = cv2.grabCut(image, mask, rect=None, bgdModel=bgd,
                                     fgdModel=fgd, iterCount=5,
                                     mode=cv2.GC_INIT_WITH_MASK)
        self.grab_cut_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(
            'uint8') * 255

    def execute(self, image):
        self.ostuSegmentation(image)
        self.grabCut(image)

        return self.thresh_img, self.grab_cut_mask

# img_path = os.path.join(conf.items("Path")[0][1], "600.jpg")
# img = cv2.imread(img_path)
# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title("Original")
#
# my_image_seg = FlowerSegmentation("gray")
# img_ostu, img_grab_cut = my_image_seg.execute(img)
# plt.subplot(1, 3, 2)
# img_ostu = img_ostu[:, :, np.newaxis]
# plt.imshow(cv2.cvtColor(img_ostu, cv2.COLOR_BGR2RGB))
# plt.title("OSTU")
#
# plt.subplot(1, 3, 3)
# img_grab_cut = img_grab_cut[:, :, np.newaxis]
# plt.imshow(cv2.cvtColor(img_grab_cut, cv2.COLOR_BGR2RGB))
# plt.title("grab Cut")

def show(img, title):
    img = img[:, :, np.newaxis]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)

fig = plt.figure(figsize=(9, 100))
my_image_seg = FlowerSegmentation("gray")
for i in range(99):
    img_path = os.path.join(conf.items("Path")[0][1], str(600+i) + ".jpg")
    img = cv2.imread(img_path)
    img_ostu, img_grab_cut = my_image_seg.execute(img)
    plt.subplot(100, 3, 3 * i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(100, 3, 3 * i + 2)
    show(img_ostu, "OSTU")
    plt.subplot(100, 3, 3 * i + 3)
    show(img_grab_cut, "Grab Cut")

plt.subplots_adjust(hspace=1.3)
plt.savefig("plots.svg", format='svg', dpi=1600, bbox_inches='tight')
plt.show()
