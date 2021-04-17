# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：boundContours.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：17/04/2021 09:04 
"""
import cv2
import numpy as np

img = cv2.pyrDown(cv2.imread("../../images/Dress.JPG", cv2.IMREAD_UNCHANGED))

ret, thresh = cv2.threshold(src=cv2.cvtColor(src=img.copy(),
                                             code=cv2.COLOR_BGR2GRAY),
                            thresh=127, maxval=255, type=cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # Find bounding box coordinates
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0),
                  thickness=2)

    # Find minimum area
    rect = cv2.minAreaRect(c)
    # Calculate coordinate of teh minimum area rectangle
    box = cv2.boxPoints(box=rect)
    # Normalise coordinate to integers
    box = np.int0(box)
    # Draw contours
    cv2.drawContours(image=img, contours=[box], contourIdx=0,
                     color=(0, 255, 0), thickness=3)

    # Calculate center and radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    # cast to integers
    center = (int(x), int(y))
    radius = int(radius)
    # draw the circle
    img = cv2.circle(img=img, center=center, radius=radius, color=(0, 255, 0),
                     thickness=2)

cv2.drawContours(image=img, contours=contours, contourIdx=-1,
                 color=(255, 0, 0), thickness=1)
cv2.imshow("contours", img)
cv2.waitKey()
cv2.destroyAllWindows()