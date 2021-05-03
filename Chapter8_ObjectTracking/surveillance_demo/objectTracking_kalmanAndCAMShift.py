# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：objectTracking_kalmanAndCAMShift.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：01/05/2021 17:04 
"""
__version__ = "0.0.1"
__status__ = "Development"

import cv2
import configparser
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm",
                    help="m (or nothing) for MeanShift and c for CAMShift")
args = vars(parser.parse_args())

dir_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(dir_path, "config.ini")
conf = configparser.ConfigParser()
conf.read(config_path, encoding='utf-8')


def center(points):
    """Calculates the centroid of a given matrix."""
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[1][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)


font = cv2.FONT_HERSHEY_SIMPLEX


class Pedestrian():
    """Pedestrian class.

    Each pedestrian is composed of a ROI, an ID and a Kalman filter, so we
    create a pedestrian class to hold the object state.
    """

    def __init__(self, id, frame, track_window):
        """Init the pedestrian object with track window coordinates."""
        # set up the roi
        self.id = int(id)
        x, y, w, h = track_window
        self.track_window = track_window
        self.roi = cv2.cvtColor(frame[y:y + h, x:x + w], code=cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist(images=[self.roi], channels=[0], mask=None,
                                histSize=[16], ranges=[0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX)

        # setup the kalman
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]],
                                                 np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32)
        self.kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32) * 0.03
        self.measurement = np.array((2, 1), np.float32)
        self.prediction = np.zeros((2, 1), np.float32)
        self.term_crit = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None
        self.update(frame)

    def __del__(self):
        print("Pedestrian {:d} destroyed!".format(self.id))

    def update(self, frame):
        print("Updating: {:d}".format(self.id))
        hsv = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
        back_project = cv2.calcBackProject(images=[hsv], channels=[0],
                                           hist=self.roi_hist, ranges=[0, 180],
                                           scale=1)

        if args.get("algorithm") == "c":
            ret, self.track_window = cv2.CamShift(probImage=back_project,
                                                  window=self.track_window,
                                                  criteria=self.term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            self.center = center(pts)
            cv2.polylines(frame, pts=[pts], isClosed=True, color=(0, 255, 0),
                          thickness=1)

        if not args.get("algorithm") or args.get("algorithm") == "m":
            ret, self.track_window = cv2.meanShift(probImage=back_project,
                                                   window=self.track_window,
                                                   criteria=self.term_crit)
            x, y, w, h = self.track_window
            self.center = center(
                [[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        self.kalman.correct(self.center)
        prediction = self.kalman.predict()
        cv2.circle(frame, center=(int(prediction[0]), int(prediction[1])),
                   radius=4, color=(255, 0, 0), thickness=-1)

        # fake shadow
        cv2.putText(frame, "ID: {:d} -> {:s}".format(self.id, self.center),
                    org=(11, (self.id + 1) * 25 + 1), fontFace=font,
                    fontScale=0.6, color=(0, 0, 0), thickness=1,
                    lineType=cv2.LINE_AA)


def main():
    video = cv2.VideoCapture(conf.items("Path")[0][1])
    history = 20
    bg_subtractor_knn = cv2.createBackgroundSubtractorKNN()

    cv2.namedWindow("Surveillance")
    pedestrians = {}
    firstFrame = True
    frames = 0
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("ooutput.avi", fourcc, 20.0, (640, 480))
    success, frame = video.read()
    while success:
        print(
            "------------------------- FRAME {:d} ---------------------".format(
                frames))

        fgmask = bg_subtractor_knn.apply(frame)

        # This is just to let the background subtractor build a bit of history
        if frames < history:
            frames += 1
            continue

        _, thresh = cv2.threshold(fgmask.copy(), thresh=127, maxval=255,
                               type=cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh,
                           kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                            (3, 3)),
                           iterations=3)
        dilated = cv2.dilate(thresh,
                             kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                              (8, 3)),
                             iterations=2)
        contours, hierarchy = cv2.findContours(dilated, mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_SIMPLE)

        counter = 0
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # only create pedestrians in the first frame, then just follow the ones you have
                if firstFrame is True:
                    pedestrians[contour] = Pedestrian(counter, frame,
                                                      (x, y, w, h))
                counter += 1

        for i, p in pedestrians.items():
            p.update(frame)

        firstFrame = False
        frames += 1

        cv2.imshow("Surveillance", frame)
        out.write(frame)
        if cv2.waitKey(10) & 0xff == ord("q"):
            break
        success, frame = video.read()
    out.release()


main()
# if __name__ == "__main__":
#     main()
