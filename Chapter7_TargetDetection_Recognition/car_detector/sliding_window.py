# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：sliding_window.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：29/04/2021 14:25 
"""
def slidingWindow(img, step_size, window_size):
    """Sliding window: This is used to detecting the same type objects in one
    figure, namely, the object number can be more than one. This function
    will return a generator.
    -------------
    :param window_size (100, 40): 40 is the height, 100 is the width
    :return
    @generator structure: (x, y, img_window)
    x, y are the upper left coordinates of the specific sliding window.
    img_window is the ROI.
    """
    #
    for y in range(0, img.shape[0], step_size):
        for x in range(0, img.shape[1], step_size):
            yield x, y, img[y:y + window_size[1], x:x + window_size[0]]


