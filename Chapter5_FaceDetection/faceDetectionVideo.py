# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：faceDetectionVideo.py.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：22/04/2021 17:59 
"""
import cv2

filename = "../../images/Face.JPG"
face_detect = cv2.CascadeClassifier(
    "./cascades/haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(filename)
success, frame = video.read()
while success:
    # do detection
    img = cv2.pyrDown(frame)
    gray_src = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray_src, scaleFactor=1.3,
                                         minNeighbors=5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # show the detection
    cv2.imshow("Face Detection Video", img)
    if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
        break

    success, frame = video.read()

cv2.destroyAllWindows()
