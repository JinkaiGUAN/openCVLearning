# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：circleDetect.py.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：17/04/2021 10:31 
"""
import cv2
import numpy as np

planet_img = cv2.imread("../../images/planet.jpg", cv2.IMREAD_UNCHANGED)
gray_img = cv2.cvtColor(src=planet_img, code=cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray_img, ksize=5)
cimg = cv2.cvtColor(img, code=cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=120,
                           param1=100, param2=30, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(planet_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(planet_img, center=(i[0], i[1]), radius=2, color=(0, 0, 255),
               thickness=3)

cv2.imwrite("planet_circles.jpg", planet_img)
cv2.imshow("HoughCircles", planet_img)
cv2.waitKey()
cv2.destroyAllWindows()

