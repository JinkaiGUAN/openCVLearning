# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：filters.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：11/04/2021 13:25 
"""
import cv2
import numpy as np


class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR."""

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""

    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    """A blur filter with a 2-pixel radius."""

    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""

    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)


def strokeEdges(src, dst, blur_ksize=7, edge_ksize=5):
    if blur_ksize >= 3:
        # Before spotting the edge, use the blurred operation for the image
        blurred_src = cv2.medianBlur(src, blur_ksize)
        gray_src = cv2.cvtColor(blurred_src, cv2.COLOR_BGR2GRAY)
    else:
        # Close the blur effect
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray_src, cv2.CV_8U, gray_src, ksize=edge_ksize)
    normalized_inverse_alpha = (1.0 / 255) * (255 - gray_src)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalized_inverse_alpha
    cv2.merge(channels, dst)


if __name__ == "__main__":
    # img = np.zeros((200, 200), dtype=np.uint8)
    # img[50:150, 50:150] = 255
    img = cv2.imread("../../images/Dress.JPG", 0)

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", color)
    cv2.imshow("thresh", thresh)
    print(ret)
    cv2.waitKey()
    cv2.destroyAllWindows()


