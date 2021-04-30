# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：pyramid.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：29/04/2021 13:45 
"""
import cv2


def resize(img, scale_factor):
    """
    Return a resized image source via scale_factor.
    ---------------------
    :param img: img source.
    :param scale_factor: scale factor. Normally, the value should be larger
     than one, so the return images would be smaller than before.
    :return: the resized img source
    """
    return cv2.resize(img, dsize=(int(img.shape[1] * (1 / scale_factor)),
                                  int(img.shape[0] * (1 / scale_factor))),
                      interpolation=cv2.INTER_AREA)


def pyramid(img, scale=1.5, min_size=(200, 80)):
    """
    Return a generator.
    :param img: img source.
    :param scale: pyramid scale, normally larger than one.
    :param min_size: (200, 80), 200 is the width, and 80 is the height.
    :return
    """
    yield img

    while True:
        img = resize(img, scale) # upper sampling the original image.
        if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
            break

        yield img
