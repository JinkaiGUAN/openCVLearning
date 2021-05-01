# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：mean_shift.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：30/04/2021 20:11 
"""
import cv2
import numpy as np
import configparser

conf_path = './config.ini'
conf = configparser.ConfigParser
conf.read(filenames=conf_path, encoding='utf-8')
