# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：lane_line_detection.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：05/05/2021 08:22 
"""
import os
import cv2
import matplotlib.pyplot as plt
import configparser
import numpy as np

dir_name = os.path.dirname(os.path.abspath(__file__))
conf_path = os.path.join(dir_name, "config.ini")
conf = configparser.ConfigParser()
conf.read(conf_path, encoding='utf-8')


class LaneLineDetect():
    """
    Simple lane line detector for images.

    """

    def __init__(self, conf, plot_img=True):
        self.canny_params = conf.items("Canny")
        self.hough_params = conf.items("Hough_Lines")
        self.draw_line = plot_img

    def filterLines(self, lines, angle):
        """
        Filter the lines with large angle

        :param lines: The list of lines detected from HoughLinesP(). It is a
         list consisting of a 2d matrix, namely, [[x1, y1, x2, y2]].
        :param angle: The max angle of a line.

        :return filter_lines: a list of lines with the same structure of @param
         lines.
        """
        filter_lines = list(filter(lambda x: np.abs(x[0][3] - x[0][1]) > np.tan(
            angle * np.pi / 180) * np.abs(x[0][2] - x[0][0]), lines))
        return filter_lines

    def calcLengthAndSlope(self, line):
        """
        Helper function to calculate length and slope of the line.

        :param line, [[x1, y1, x2, y2]], 2D matrix.
        :return length, the length of the line
        :return slope, the slope of the line
        """
        length = np.square(line[0][3] - line[0][1]) + np.square(
            line[0][2] - line[0][0])
        slope = (line[0][3] - line[0][1]) / (line[0][2] - line[0][1] + 1e-6)
        return length, slope

    def drawLines(self, img, lines, color):
        """
        Helper function to draw lines in the image according to the lines.

        :param
        """
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), color, 2)
        # return img

    def groupLines(self, lines, merge_angle):
        """
        Kills off redundant lines.

        :param lines: The list of lines detected from HoughLinesP(). It is a
         list consisting of a 2d matrix, namely, [[x1, y1, x2, y2]].
        :param merge_angle: If the angle between two lines is less than this
         value, we will merge them.
        """
        lines = np.asarray(lines)
        slopes = np.array(list(map(lambda line: (line[0][3] - line[0][1]) / (
                line[0][2] - line[0][0] + 1e-6), lines)))
        print("type of slopes: {}".format(slopes.dtype))
        # ranking the slopes
        sorted_lines = list(lines[np.argsort(slopes)])

        merge_slope = float(merge_angle) * np.pi / 180.0

        if len(sorted_lines) > 0:
            loop_mark = True
            ini_line = sorted_lines[0]
            i = 1
            ini_length, ini_slope = self.calcLengthAndSlope(ini_line)
        else:
            loop_mark = False

        print("sorted line length", len(sorted_lines))
        while loop_mark: # len(sorted_lines):
            # i is the number of lines in the sorted_lines
            if i >= len(sorted_lines):
                break

            cur_line = sorted_lines[i]
            cur_length, cur_slope = self.calcLengthAndSlope(cur_line)
            if np.abs(cur_slope - ini_slope) < merge_slope:
                if cur_length > ini_length:
                    ini_slope = cur_slope
                    ini_length = cur_length
                    sorted_lines.pop(i - 1)
                else:
                    sorted_lines.pop(i)
                i -= 1
            else:
                ini_slope = cur_slope
                ini_length = cur_length
            i += 1

        return sorted_lines

    def execute(self, img):
        """
        Main program to detect the lane line in a single pic.

        :param img: A BGR image or frame.
        :return:
        """

        # 1. Restrict the ROI, only the bottom half can be used
        height, width, channel = img.shape
        xmin, xmax, ymin, ymax = 0, width, int(height * 3 / 5), height

        # 2. Canny/Edge detection
        img_gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(img_gray, ksize=(
            int(self.canny_params[2][1]), int(self.canny_params[2][1])),
                                        sigmaX=0, sigmaY=0)
        edges = cv2.Canny(img_gaussian[ymin:ymax, xmin:xmax],
                          threshold1=int(self.canny_params[0][1]),
                          threshold2=int(self.canny_params[1][1]),
                          apertureSize=3)
        print("Edge shape {}".format(edges.shape))
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[ymin:ymax, xmin:xmax] = edges

        if self.draw_line:
            plt.subplot(2, 4, 1)
            plt.imshow(mask, cmap="gray")

        # 3. Line detection using HoughLinesP
        lines = cv2.HoughLinesP(mask, rho=1.0, theta=np.pi / 180,
                                threshold=int(self.hough_params[3][1]),
                                minLineLength=int(self.hough_params[1][1]),
                                maxLineGap=int(self.hough_params[2][1]))
        print("Line shape {}".format(lines.shape))

        # split the lane lines into two kinds in light of the angle
        left_lines, right_lines = [], []
        for line in lines:
            # the line is a 1*4 shape 2D matrix
            for (x1, y1, x2, y2) in line:
                # careful here, y2 < y1 if we set pt2 is upper than pt1
                k = float(y2 - y1) / float(x2 - x1 + 1e-6)
                if k < 0:
                    left_lines.append(line)
                else:
                    right_lines.append(line)

        if self.draw_line:
            img_all_lines = img.copy()
            self.drawLines(img_all_lines, left_lines, (0, 0, 255))
            self.drawLines(img_all_lines, right_lines, (0, 255, 0))
            plt.subplot(2, 2, 2)
            b, g, r = cv2.split(img_all_lines)
            img_all_lines = cv2.merge([r, g, b])
            plt.imshow(img_all_lines)
            plt.title("All lines")

        # 4. Filter redundant lines.
        print("Before filter, the length of left lines {}".format(
            len(left_lines)))
        left_lines = self.filterLines(left_lines, 10)
        left_lines = self.groupLines(left_lines, 5)
        print("After filter, the length of left lines {}".format(
            len(left_lines)))
        print("Before filter, the length of right lines {}".format(
            len(right_lines)))
        right_lines = self.filterLines(right_lines, 10)
        right_lines = self.groupLines(right_lines, 5)
        print("After filter, the length of right lines {}".format(
            len(right_lines)))

        # 5. Draw the lines in the picture
        self.drawLines(img, left_lines, (0, 0, 255))
        self.drawLines(img, right_lines, (0, 255, 0))
        if self.draw_line:
            plt.subplot(2, 2, 3)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            plt.imshow(img)
            plt.title("Filter lines")
            plt.show()
        return img


lane_line_detector = LaneLineDetect(conf, plot_img=False)
img_path = conf.items("Path")[0][1]
image = cv2.imread(img_path)
# lane_line_detector.execute(image)
video_path = conf.items("Path")[1][1]
video = cv2.VideoCapture(video_path)
success, frame = video.read()
fps = int(video.get(cv2.CAP_PROP_FPS))

while success:
    frame = lane_line_detector.execute(frame)
    cv2.imshow("Video Detector", frame)
    if cv2.waitKey(fps) & 0xff == ord("q"):
        break
    success, frame = video.read()

cv2.destroyAllWindows()
