# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：generateData.py.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：23/04/2021 18:04 
"""
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class GenerateData(object):
    """
        csv_paths: The list contains all the csv files.
    """

    def __init__(self):
        self.face_detect = cv2.CascadeClassifier(
            "./cascades/haarcascade_frontalface_default.xml")
        self.video_base_path = 'E:/CV/openCV/pictureProcessBasis/images/'
        self.data_base_path = 'E:/CV/openCV/pictureProcessBasis/images/data/'
        self.csv_base_path = 'E:/CV/openCV/pictureProcessBasis/images/data/'
        self.person_name = ['Jobs', 'Musk', 'Andrew']
        self.video_path = []
        self.data_path = []
        self.csv_paths = []

    def getPersonName(self):
        return self.person_name

    def getCsvPath(self):
        return self.csv_paths

    def generatePath(self):
        self.video_path = [os.path.join(self.video_base_path, name + '.mp4')
                           for name in self.person_name]
        self.data_path = [os.path.join(self.data_base_path, name + '/') for
                          name in self.person_name]

    def retrievePicsFromVideo(self, video, save_path):
        # Check the folder
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        image_list = []
        success, frame = video.read()
        count = 0
        while success:
            frame = cv2.pyrDown(frame)
            gray_frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
            faces = self.face_detect.detectMultiScale(gray_frame,
                                                      scaleFactor=1.3,
                                                      minNeighbors=5)
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (255, 0, 0), 2)

                # roi_face = gray_frame[x:x + w, y:y + h]
                roi_face = gray_frame[y:y + h, x:x + w]
                roi_face = cv2.resize(roi_face, dsize=(200, 200))
                frame_path = os.path.join(save_path, str(count) + '.pgm')
                # Obtain the image address
                image_list.append(frame_path)
                cv2.imwrite(frame_path, roi_face)
                count += 1

            success, frame = video.read()

        return image_list

    def generatePics(self):
        for i, name in enumerate(self.person_name):
            video = cv2.VideoCapture(self.video_path[i])
            images = self.retrievePicsFromVideo(video, self.data_path[i])
            csv_path = os.path.join(self.csv_base_path, name + '.csv')
            training_set = pd.DataFrame({'Path': images, 'label': i})
            self.csv_paths.append(csv_path)
            training_set.to_csv(csv_path)

    def run(self):
        # generate video paths
        self.generatePath()
        # generate pics
        self.generatePics()


class FaceClassifier(object):
    """:arg
        model_name: There are three kinds of classifier used in this demo,
            namely, "EigenFaces", "LBPH", "FisherFaces".
    """

    def __init__(self, csvs, model_name="EigenFaces", pic_or_video="pic"):
        self.model_name = model_name  # "EigenFaces", "LBPH", "FisherFaces"
        self.pic_or_video = pic_or_video
        self.person_name = ['Jobs', 'Musk', 'Andrew']
        self.csv_paths = csvs
        self.face_detect = cv2.CascadeClassifier(
            "./cascades/haarcascade_frontalface_default.xml")

    def extractPics(self, sz=None):
        X, y = [], []
        for i, single_csv in enumerate(self.csv_paths):
            pics_path = pd.read_csv(single_csv)["Path"]
            for pic_path in pics_path:
                img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)

                # resize to given size (if given)
                if sz is not None:
                    img = cv2.resize(img, (200, 200))
                X.append(np.asarray(img, dtype=np.uint8))
                y.append(i)

        y = np.asarray(y, dtype=np.int32)
        return [X, y]

    def run(self):
        # EigenfacesModel
        eigen_faces_model = self.generateFacesModel()
        if self.pic_or_video == "pic":
            self.facesClassifierForPics(eigen_faces_model)
        elif self.pic_or_video == "video":
            self.facesClassifierForVideo(eigen_faces_model)
        else:
            print("Please input the valid key value for pic_or_video!")

    def generateFacesModel(self):
        [X, y] = self.extractPics()
        if self.model_name == "EigenFaces":
            model = cv2.face.EigenFaceRecognizer_create()
        elif self.model_name == "LBPH":
            model = cv2.face.LBPHFaceRecognizer_create()
        elif self.model_name == "FisherFaces":
            model = cv2.face.FisherFaceRecognizer_create()
        else:
            print("Please input the valid key for model_name!")
        model.train(X, y)

        return model

    def facesClassifierForPics(self, model):
        img = cv2.imread("../../images/Jobs.png")
        img = cv2.pyrDown(img)
        gray_img = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
        faces = self.face_detect.detectMultiScale(gray_img, scaleFactor=1.3,
                                                  minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_face = gray_img[y:y + h, x:x + w]
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            try:
                roi_face = cv2.resize(roi_face, dsize=(200, 200),
                                      interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi_face)
                print("Label: {}, Confidence: {}".format(params[0], params[1]))
                cv2.putText(img, self.person_name[params[0]],
                            (x, y - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 0, 0), thickness=1)
            except:
                continue
        cv2.imshow("Classifier", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def facesClassifierForVideo(self, model):
        video = cv2.VideoCapture("../../images/Jobs.mp4")
        success, frame = video.read()
        while success:
            frame = cv2.pyrDown(frame)
            gray_frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
            faces = self.face_detect.detectMultiScale(gray_frame,
                                                      scaleFactor=1.3,
                                                      minNeighbors=5)
            for (x, y, w, h) in faces:
                roi_face = gray_frame[y:y + h, x:x + w]
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (255, 0, 0), 2)
                try:
                    roi_face = cv2.resize(roi_face, dsize=(200, 200),
                                          interpolation=cv2.INTER_LINEAR)
                    params = model.predict(roi_face)
                    print("Label: {}, Confidence: {}".format(params[0],
                                                             params[1]))
                    cv2.putText(frame, self.person_name[params[0]],
                                (x, y - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=(255, 0, 0), thickness=1)
                except:
                    continue
            cv2.imshow("Classifier Video", frame)
            if cv2.waitKey(1) and 0xff == ord('q'):
                break
            success, frame = video.read()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_pics_object = GenerateData()
    # generate_pics_object.run()
    csvs = ['E:/CV/openCV/pictureProcessBasis/images/data/Jobs.csv',
            'E:/CV/openCV/pictureProcessBasis/images/data/Musk.csv',
            'E:/CV/openCV/pictureProcessBasis/images/data/Andrew.csv']
    # csvs = generate_pics_object.csv_paths
    for model_name in ["EigenFaces", "LBPH", "FisherFaces"]:
        face_classifier_object = FaceClassifier(csvs, model_name, )
        face_classifier_object.run()
