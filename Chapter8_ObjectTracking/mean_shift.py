# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：mean_shift.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：30/04/2021 20:11 
"""
import cv2
import numpy as np
import configparser

conf_path = './config.ini'
conf = configparser.ConfigParser()
conf.read(filenames=conf_path, encoding='utf-8')

video = cv2.VideoCapture(conf.items("Path")[0][1])
success, frame = video.read()

r, h, c, w = 10, 200, 10, 200
track_window = (c, r, w, h)
roi = frame[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, lowerb=np.array((100., 30., 32.)),
                   upperb=np.array((180., 120., 255.)))
roi_hist = cv2.calcHist([hsv_roi], channels=[0], mask=mask, histSize=[180],
                        ranges=[0, 180])
cv2.normalize(roi_hist, dst=roi_hist, alpha=0, beta=255,
              norm_type=cv2.NORM_MINMAX)
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while success:
    hsv = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject(images=[hsv], channels=[0], hist=roi_hist,
                              ranges=[0, 180], scale=1)
    # apply mean-shift to get the new location
    ret, track_window = cv2.meanShift(probImage=dst, window=track_window,
                                      criteria=term_crit)
    # Draw it on image
    x, y, w, h = track_window
    img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("img2", img2)
    if cv2.waitKey(30) & 0xff == ord('q'):
        break
    success, frame = video.read()

cv2.destroyAllWindows()