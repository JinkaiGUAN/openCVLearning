# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：faceDetectionVideo.py.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：22/04/2021 17:59 
"""
import cv2

filename = "../../images/faceVideo.mp4"
face_detect = cv2.CascadeClassifier(
    "./cascades/haarcascade_frontalface_default.xml")
eye_detect = cv2.CascadeClassifier("./cascades/haarcascade_eye.xml")

video = cv2.VideoCapture(filename)
fps = video.get(cv2.CAP_PROP_FPS)
success, frame = video.read()
while success:
    frame = cv2.pyrDown(frame)
    gray_frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray_frame, scaleFactor=1.3,
                                         minNeighbors=5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # change the region of interest
        h1 = int(float(h / 1.5))
        gray_roi = gray_frame[y:y + h1, x:x + w]
        eyes = eye_detect.detectMultiScale(gray_roi, scaleFactor=1.03,
                                           minNeighbors=5,
                                           flags=cv2.CASCADE_SCALE_IMAGE,
                                           minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            frame = cv2.rectangle(frame, pt1=(x + ex, y + ey),
                                  pt2=(x + ex + ew, y + ey + eh),
                                  color=(0, 255, 0), thickness=2)

    cv2.imshow("Face Detection Video", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    success, frame = video.read()

cv2.destroyAllWindows()
