# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：faceDetection.py.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：22/04/2021 16:55 
"""
import cv2

filename = "../../images/face.png"


def detect(filename):
    face_cascade = cv2.CascadeClassifier(
        "./cascades/haarcascade_frontalface_default.xml")
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.namedWindow("Face Detection")
    cv2.imshow("Face Detection", img)
    cv2.imwrite('./faceDetection.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect(filename)



