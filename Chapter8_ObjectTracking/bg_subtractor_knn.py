# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning
@File    ：bg_subtractor_knn.py
@IDE     ：PyCharm
@Author  ：Peter
@Date    ：30/04/2021 19:14
"""
import cv2
import numpy as np
import configparser

conf_path = './config.ini'
conf = configparser.ConfigParser() # create key-value management object
conf.read(filenames=conf_path, encoding='utf-8')
# video_path = conf.items("Path")[0][1]

# video_path = "./movie.mpg"
video = cv2.VideoCapture(video_path = conf.items("Path")[0][1])
bg_knn = cv2.createBackgroundSubtractorKNN(detectShadows=True)
success, frame = video.read()
fps = video.get(cv2.CAP_PROP_FPS)

while success:
    fgmask = bg_knn.apply(frame)
    mask_thresh = cv2.threshold(fgmask.copy(), thresh=244, maxval=255,
                                type=cv2.THRESH_BINARY)[1] #会将低于该阈值的变为0， 高于该阈值会变为1.
    mask_dilated = cv2.dilate(mask_thresh, kernel=cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, ksize=(3, 3)), iterations=2)

    contours, hierarchy = cv2.findContours(mask_dilated, mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1600:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow("knn", fgmask)
    cv2.imshow("thresh", mask_thresh)
    cv2.imshow("detection", frame)
    if cv2.waitKey(int(fps)) & 0xff == ord('q'):
        break

    success, frame = video.read()

cv2.destroyAllWindows()
