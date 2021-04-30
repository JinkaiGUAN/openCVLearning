# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：non_maximum.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：29/04/2021 14:34 
"""
import numpy as np


def area(box):
    """Calculate the are of the box that is a list [x1, y1, x2, y2], i.e.
    the upper left and bottom right points."""
    return (box[2] - box[1]) * (box[3] - box[0])


def overlap(a, b, threshold=0.5):
    """Return whether the overlapping are is large or not!
    ---------------
    a, b: the list of teh coordinates for boxes a and b."""
    print("Checking overlap for {} and {}".format(a, b))
    x1 = np.maximum(a[0], b[0])
    y1 = np.maximum(a[1], b[1])
    x2 = np.minimum(a[2], b[2])
    y2 = np.minimum(a[3], b[3])
    inter_set = float(area([x1, y1, x2, y2]))
    return inter_set / np.minimum(area(a), area(b)) >= threshold


def is_inside(a, b):
    """Checking whether a is in b or b is in a."""

    def inside(rect1, rect2):
        """Check the rect1 is in the rect2 or not.
        True means yes, otherwise, false"""
        if rect1[0] > rect2[0] and rect1[2] < rect2[2]:
            return rect1[1] > rect2[1] and rect1[3] < rect2[3]
        else:
            return False

    return inside(rect1=a, rect2=b) or inside(rect1=b, rect2=b)


def nomMaximumSuppressionFast(boxes, overlap_threshold=0.5):
    """
    The non maximum suppression algorithm.
    ----------------------
    :param boxes:np.array([[x1, y1, x2, y2, score], ....]), (x1, y1) is the
     upper left point of the box, (x2, y2) is the bottom right point of the box.
     score is the positive value.
    :return boxes: the preferable boxes after filtering.
    """
    # If there is no boxes, return an empty list
    if len(boxes) == 0:
        return []

    scores = boxes[:, 4]
    scores_idx = np.argsort(scores)

    while len(scores_idx) > 0:
        # ref_box = scores[scores_idx[0]]
        ref_box = scores_idx[0]
        print("Checking box")
        to_delete = []
        for idx in scores_idx:
            if idx == 0:
                continue
            try:
                # if the overlapping area is larger than the threshold,
                # we should drop it off.
                if (overlap(boxes[idx], boxes[ref_box], overlap_threshold)):
                    to_delete.append(idx)
                    scores_idx = np.delete(scores_idx, [idx], 0)
            except:
                print("NMS wronging!")

        # delete the reference box
        boxes = np.delete(boxes, to_delete, 0)
        scores_idx = np.delete(scores_idx, 0, 0)

    return boxes
