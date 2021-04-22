# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：faceDetectionImage.py.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：22/04/2021 22:06 
"""
import cv2

filename = '../../images/Face.JPG'
# filename = "../../images/face.png"

img = cv2.imread(filename)
img = cv2.pyrDown(img)
gray_src = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)

face_detect = cv2.CascadeClassifier(
    "./cascades/haarcascade_frontalface_default.xml")
eye_detect = cv2.CascadeClassifier("./cascades/haarcascade_eye.xml")
faces = face_detect.detectMultiScale(gray_src, scaleFactor=1.3, minNeighbors=5)
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0),
                        thickness=2)
    # gain the certain region of interest for detecting eyes
    h1 = int(float(h / 1.5))
    gray_roi = gray_src[int(y):int(y + h1), int(x):int(x + w)]
    eyes = eye_detect.detectMultiScale(gray_roi, scaleFactor=1.03,
                                       minNeighbors=5,
                                       flags=cv2.CASCADE_SCALE_IMAGE,
                                       minSize=(20, 20))
    for (ex, ey, ew, eh) in eyes:
        # attention needed, we need to add the coordinate-shift
        img = cv2.rectangle(img, pt1=(x+ex, y+ey), pt2=(x+ex + ew, y+ey + eh),
                            color=(0, 255, 0), thickness=2)

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
