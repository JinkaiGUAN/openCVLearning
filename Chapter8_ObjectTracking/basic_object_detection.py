# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：basic_object_detection.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：30/04/2021 17:01 
"""
import cv2
import numpy as np

movie_path = "movie.mpg"
video = cv2.VideoCapture(movie_path)
success, frame = video.read()

es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(10,10))
kernel = np.ones(shape=(5, 5), dtype=np.uint8)
background = None

while success:
    if background is None:
        background = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(background, ksize=(21, 21), sigmaX=0)
        continue

    gray_frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(background, ksize=(21, 21), sigmaX=0)

    diff = cv2.absdiff(background, gray_frame)
    diff = cv2.threshold(diff, thresh=15, maxval=255, type=cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, es, iterations=2)
    cnts, hierarchy = cv2.findContours(diff.copy(),
                                       mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        if cv2.contourArea(cnt) < 1500:
            continue
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Contours", frame)
    cv2.imshow("diff", diff)
    if cv2.waitKey(10) & 0xff == ord("q"):
        break
    success, frame = video.read()

cv2.destroyAllWindows()