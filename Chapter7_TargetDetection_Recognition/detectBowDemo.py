# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：detectBowDemo.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：29/04/2021 20:44 
"""
import cv2
import numpy as np

data_path = "./CarData/TrainImages"
pos, neg = "pos-", "neg-"
DESCRIPTOR_NUM = 40
TRAINING_NUM = 400

def path(cls, i):
    """
    @cls: The class of the image, i.e. pos or neg
    @i: The number of the image, i.e. 1, 2 etc.
    """
    return "{:s}/{:s}{:d}.pgm".format(data_path, cls, i)


"""1. Preparation"""
# create the SIFT
extract, detect = cv2.SIFT_create(), cv2.SIFT_create()

# creater matcher
flann_params = dict(algorithm=1, trees=5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

# create BOW trainer
bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
# Initialise the BOW extractor
extractor_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)

"""2.Clustering"""
# 2.1 Get the descriptors of each image.
def descriptor_sift(img_path):
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    return extract.compute(img, detect.detect(img))[1] # Return an array,
    # i.e. descriptor


# 2.2 Add the descriptors of each image into the BOW trainer.
for i in range(DESCRIPTOR_NUM):
    # Add positive descriptors to the trainer
    bow_kmeans_trainer.add(descriptor_sift(path(pos, i)))
    # Add negative descriptors into the trainer
    bow_kmeans_trainer.add(descriptor_sift(path(neg, i)))

# 2.3 Cluster the descriptors added into the trainer to K clusters. The
# center of the cluster is the visual word.
dictionary = bow_kmeans_trainer.cluster()
extractor_bow.setVocabulary(dictionary)

"""3. Obtaining the histogram of each image."""
# Getting the descriptors of each image using BOW extractor, then store them
# into a list, namely, training set for X. As for the training set for y, you
# need to add 1 for positive samples and -1 for negative samples.

training_set, training_label = [], []


def descriptor_bow(img_path):
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
    return extractor_bow.compute(img, detect.detect(img)) # here it only return
    # the descriptors when employing compute method


for i in range(TRAINING_NUM):
    try:
        # Add positive descriptors gotten from BOW extractor into training set
        training_set.extend(descriptor_bow(path(pos, i)))
        training_label.append(1)
        # Add negative descriptors gotten from BOW extractor into training set
        training_set.extend(descriptor_bow(path(neg, i))) # 130 is nontype
        training_label.append(-1)
    except:
        pass

"""4. Prediction: train the training set via SVM and make predictions
for new images."""
# Train via SVM
svm = cv2.ml.SVM_create()
svm.train(np.array(training_set), cv2.ml.ROW_SAMPLE, np.array(training_label))


# Make predictions
def predict(img_path):
    descriptors = descriptor_bow(img_path)
    p = svm.predict(descriptors)
    print(img_path, "\t", p[1][0][0])
    return p


car_path, nocar_path = "./car.png", "./nocar.png"
car_img = cv2.imread(car_path, cv2.IMREAD_GRAYSCALE)
nocar_img = cv2.imread(nocar_path, cv2.IMREAD_GRAYSCALE)
car_predict = predict(car_path)
nocar_predict = predict(nocar_path)

font = cv2.FONT_HERSHEY_SIMPLEX
if car_predict[1][0][0] == 1.0:
    cv2.putText(car_img, "Car detected", org=(10, 30), fontFace=font,
                fontScale=1, color=(0, 255, 0), thickness=2,
                lineType=cv2.LINE_AA)
if nocar_predict[1][0][0] == -1.0:
    cv2.putText(nocar_img, "Car not detected", org=(10, 30), fontFace=font,
                fontScale=1, color=(0, 255, 0), thickness=2,
                lineType=cv2.LINE_AA)

cv2.imshow("BOW + SVM Success", car_img)
cv2.imshow("BOW + SYM Failure", nocar_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
