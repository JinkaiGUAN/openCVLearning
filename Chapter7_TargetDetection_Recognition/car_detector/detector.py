# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：detector.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：29/04/2021 14:58 
"""
import cv2
import configparser
import numpy as np

config_path = "E:\CV\openCV\pictureProcessBasis\openCVLearning\Chapter7_TargetDetection_Recognition\config.ini"
conf = configparser.ConfigParser()  # create key-value management object
conf.read(config_path, encoding='utf-8')

# data_path = conf.items("Path")[0][1]
# FEATURE_SAMPLES = int(conf.items("Sampling Data")[0][1])
# TRAINING_SAMPLES = int(conf.items("Sampling Data")[1][1])


# data_path = './CarData/TrainImages'  # The path here is the relative path
# to the total main python file.
# FEATURE_SAMPLES = 100
# TRAINING_SAMPLES = 400


def path(cls, i):
    """Return path"""
    return "{:s}/{:s}{:d}.pgm".format(conf.items("Path")[0][1], cls, i + 1)


def getFlannMatcher():
    """Getting the FLANN matcher."""
    flann_params = dict(algorithm=1, trees=5)
    return cv2.FlannBasedMatcher(flann_params, {})


def getBowExtractor(extract, flann):
    """Get the BOW descriptor extractor from the corresponding extractor
     (normally, SIFT) and flann matcher."""
    return cv2.BOWImgDescriptorExtractor(extract, flann)


def getExtractAndDetect():
    """Get the SIFT extractors and detectors."""
    return cv2.SIFT_create(), cv2.SIFT_create()


def descriptorSift(fn, extractor, detector):
    """Compute the descriptors of the image.
    --------------
    :argument
    @fn: the image path
    """
    img = cv2.imread(fn, flags=cv2.IMREAD_GRAYSCALE)
    return extractor.compute(img, detector.detect(img))[1]


def bowFeatures(img, extractor_bow, detector):
    """Compute the descriptors of the image.
    -----------------
    :argument
    @img: the BGR img source.
    """
    return extractor_bow.compute(img, detector.detect(img))


def carDetector():
    """It is the function than use BOW extractor to give the method to train
     data.
    --------------
    :return
    @svm: The trained SVM tool that can be used to make some predictions.
    @extract_bow: The BOW extractor, used to produce the descriptors of the
     corresponding images and key-points. The key-points can be gotten from
     the SIFT detector, using detect method.
    """
    # 1. Create the SIFT detectors and extractors
    pos, neg = "pos-", "neg-"
    detect, extract = getExtractAndDetect()
    matcher = getFlannMatcher()
    print("Building BOWKMeansTrainer...")
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
    extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)

    # 2. Cluster
    print("Adding features/descriptors to the trainer.")
    for i in range(int(conf.items("Sampling Data")[0][1])):
        # try:
        bow_kmeans_trainer.add(descriptorSift(path(pos, i), extract, detect))
        bow_kmeans_trainer.add(descriptorSift(path(neg, i), extract, detect))
        # except:
        #     pass

    vocabulary = bow_kmeans_trainer.cluster()

    extract_bow.setVocabulary(vocabulary)

    # 3. Obtain the histogram (i.e. training data) of each image.
    train_data, train_labels = [], []
    print("Adding to train data.")
    for i in range(int(conf.items("Sampling Data")[1][1])):
        try:
            train_data.extend(
                bowFeatures(
                    cv2.imread(path(pos, i), flags=cv2.IMREAD_GRAYSCALE),
                    extract_bow, detect))
            train_labels.append(1)
            train_data.extend(
                bowFeatures(
                    cv2.imread(path(neg, i), flags=cv2.IMREAD_GRAYSCALE),
                    extract_bow, detect))
            train_labels.append(-1)
        except:
            pass

    # 4. Train the SVM
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(0.5)
    svm.setC(30)
    svm.setKernel(cv2.ml.SVM_RBF)

    svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_labels))
    return svm, extract_bow


if __name__ == "__main__":
    img_path = '../car.png'
    img = cv2.imread(img_path, 0)
    _, detector = getExtractAndDetect()
    svm, extract_bow = carDetector()
    descriptors = extract_bow.compute(img, keypoints=detector.detect(img))
    p = svm.predict(descriptors)
    print(p)
