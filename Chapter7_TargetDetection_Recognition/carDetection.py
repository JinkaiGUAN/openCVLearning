# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：carDetection.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：29/04/2021 13:35
@Description: The scale in this project is width/height.
"""
import cv2
import numpy as np
from car_detector.detector import carDetector, bowFeatures
from car_detector.pyramid import pyramid
from car_detector.non_maximum import nomMaximumSuppressionFast as nms
from car_detector.sliding_window import slidingWindow


# test_img_path = "./CarData/TestImages"
# img_path = './CarData/TestImages/test-0.pgm'
img_path = './car.png'
# img_path = "./CarData/TestImages_Scale/test-0.pgm"

svm, extractor = carDetector()
detect = cv2.SIFT_create()

# w, h = 100, 40
w, h = 600, 250
img = cv2.imread(img_path)


rectangles = []
counter = 1
scaleFactor = 1.25
scale = 1
font = cv2.FONT_HERSHEY_SIMPLEX
scores = []

for resized in pyramid(img, scaleFactor):
    # Obtain the scale of the resized image.
    scale = float(img.shape[1]) / float(resized.shape[1])
    for (x, y, roi) in slidingWindow(resized, step_size=10,
                                     window_size=(w, h)):
        # print(roi.shape)
        if roi.shape[1] != w or roi.shape[0] != h:
            # print("Delete!")
            continue

        try:
            descriptors = bowFeatures(img=roi, extractor_bow=extractor,
                                      detector=detect)
            _, result = svm.predict(descriptors)
            a, res = svm.predict(descriptors,
                                 flags=cv2.ml.STAT_MODEL_RAW_OUTPUT | cv2.ml.STAT_MODEL_UPDATE_MODEL)
            print("Class: {}, Score: {}, a: {}".format(result[0][0],
                                                       res[0][0], res))
            score = res[0][0]
            scores.append(score)
            if result[0][0] == 1:
                if score < -0.35:
                    # all scores less than ths=is value will be treated as a good prediction.
                    rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int(
                        (x + w) * scale), int((y + h) * scale)
                    rectangles.append([rx, ry, rx2, ry2, abs(score)])
        except:
            print("The prediction went wrong!")

        counter += 1

windows = np.array(rectangles)
boxes = nms(windows, overlap_threshold=0.25)

for (x, y, x2, y2, score) in boxes:
    print(x, y, x2, y2, score)
    cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.putText(img, "{:f}".format(score), (int(x), int(y)), fontFace=font,
                fontScale=1, color=(0, 255, 0))

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
