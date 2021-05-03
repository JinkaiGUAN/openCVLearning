# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：kalMan_CAMShift.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：01/05/2021 15:15 
"""
import cv2
import numpy as np


frame = np.zeros(shape=(800, 800, 3), dtype=np.uint8)
last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)

kalman = cv2.KalmanFilter(4, 2, 1)
# 设置测量矩阵
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
# 设置过程噪声矩阵
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
# 设置过程噪声协方差矩阵
kalman.processNoiseCov = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

def mousemove(event, x, y, s, p):
    """函数在这里的作用就是传递X, Y的坐标值，便于对轨迹进行卡尔曼滤波
    """
    global frame, current_measurement, measurements, last_measurement, \
        current_measurement, current_prediction, last_prediction

    last_prediction = current_prediction
    last_measurement = current_measurement
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    kalman.correct(current_measurement)
    current_prediction = kalman.predict()
    lmx, lmy = last_measurement[0], last_measurement[1]
    cmx, cmy = current_measurement[0], current_measurement[1]
    lpx, lpy = last_prediction[0], last_prediction[1]
    cpx, cpy = current_prediction[0], current_prediction[1]
    cv2.line(frame, (int(lmx), int(lmy)), (int(cmx), int(cmy)), (0, 100, 0))
    cv2.line(frame, (int(lpx), int(lpy)), (int(cpx), int(cpy)), (0, 0, 200))


cv2.namedWindow("kalman_tracker")
cv2.setMouseCallback("kalman_tracker", mousemove)


while True:
    cv2.imshow("kalman_tracker", frame)
    if (cv2.waitKey(30) & 0xff) == 27:
        break
    if (cv2.waitKey(30) & 0xff) == ord('q'):
        cv2.imwrite("kalman.jpg", frame)
        break

cv2.destroyAllWindows()
