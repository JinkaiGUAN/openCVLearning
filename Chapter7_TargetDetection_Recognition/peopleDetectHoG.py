# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：peopleDetectHoG.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：29/04/2021 19:36
@Description: Using Hog to detect persons in a image.
"""
import cv2
import numpy as np


def isInside(a, b):
    def inside(rect1, rect2):
        if rect1[0] > rect2[0] and rect1[2] < rect2[2]:
            return rect1[1] > rect2[1] and rect1[3] < rect2[3]
        else:
            return False

    return inside(a, b) or inside(b, a)


def drawPerson(src, person_box):
    x, y, w, h = person_box
    return cv2.rectangle(src, pt1=(x, y), pt2=(x + w, y + h),
                         color=(0, 255, 0), thickness=2)


img_path = '../../images/people.png'
img = cv2.imread(img_path)
img = cv2.pyrDown(img)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

found, w = hog.detectMultiScale(img, winStride=(8, 8), scale=1.06)

found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and isInside(r, q):
            break
        else:
            found_filtered.append(r)

for person in found_filtered:
    img = drawPerson(img, person)

cv2.imshow("people detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
