# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：background_subtractor.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：30/04/2021 19:14 
"""
import cv2
import numpy as np

video_path = "./movie.mpg"
video = cv2.VideoCapture(video_path)
mog = cv2.createBackgroundSubtractorMOG2()

success, frame = video.read()
while success:
    fgmask = mog.apply(frame)
    cv2.imshow("frame", fgmask)
    if cv2.waitKey(10) & 0xff == ord("q"):
        break
    success, frame = video.read()

cv2.destroyAllWindows()