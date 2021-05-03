# -*- coding: UTF-8 -*-
"""
@Project ：openCVLearning 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：Peter
@Date    ：02/05/2021 23:25 
"""
import os
import sys
import argparse

"""
print(__file__)
print("Absolute path: {}".format(os.path.abspath(__file__)))
print("Real path: {}".format(os.path.realpath(__file__)))
print("Directory name {}".format(os.path.dirname(os.path.realpath(__file__))))
print("Path split {}".format(os.path.split(os.path.realpath(__file__))))
print("Path split {} [0]".format(os.path.split(os.path.realpath(__file__)))[0])
print("os.getcwd() = {}".format(os.getcwd()))

current_path = os.path.dirname(os.path.realpath(__file__))
fig = os.path.join(current_path, "test.py")
print(fig)

# 一般最好直接读取绝对路径。
print("Absolute path {}".format(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')))
#
"""

parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument('Integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum,
                    default=max,
                    help='sum the integers (default: find the max)')
parser.add_argument("-a", "--algorithm",
                    help="m (or nothing) for MeanShift and c for CAMShift")
args1 = vars(parser.parse_args())
args = parser.parse_args()
print(args.accumulate(args.integers))