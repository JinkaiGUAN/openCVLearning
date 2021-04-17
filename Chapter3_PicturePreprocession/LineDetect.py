# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：LineDetect.py.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：17/04/2021 09:51 
"""
import cv2
import numpy as np

img = cv2.imread("../../images/Dress.JPG", cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image=gray, threshold1=50, threshold2=120)
min_line_length = 20
max_line_Gap = 5
lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100,
                        minLineLength=min_line_length, maxLineGap=max_line_Gap)

for x1, y1, x2, y2 in lines[0]:
    cv2.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0),
             thickness=2)

cv2.imshow("edges", edges)
cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows()