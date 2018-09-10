#!/usr/bin/env python2
# -*-coding:utf-8 -*-
import cv2
import numpy as np
np.set_printoptions(precision=2)
import sys
import os
import openface
import glob
import itertools
import time
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
class Rep:
    def __init__(self):
        # モデル読み込み
        self.align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
        self.net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), 96)
    def get_rep(self,imgPath):
        # 画像の読み込み
        bgrImg = cv2.imread(imgPath)
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        # boundingboxやalign faceなどを作成
        bb = self.align.getLargestFaceBoundingBox(rgbImg)
        alignedFace = self.align.align(96, rgbImg, bb,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        # 128D vectorに変換
        rep = self.net.forward(alignedFace)
        return rep
if __name__=="__main__":
    starttime = time.time()
    reps = Rep()
    img1 = "/home/picture_mining/face_image/S__9936970.jpg.jpg"
    img1_rep = reps.get_rep(img1)
    print(time.time() - starttime)
    score_1 = 1.0
    score_2 = 1.0
    score_3 = 1.0
    img_1 = ""
    img_2 = ""
    img_3 = ""
    for i in range(0, 100):
        try:
            img2 = "/home/picture_mining/face_image_nogi/00." + str(i) + ".jpg.jpg"
            img2_rep = reps.get_rep(img2)
            d = img1_rep - img2_rep
            print(time.time() - starttime)
            print(img2)
            if score_3 > np.dot(d,d):
                if score_2 > np.dot(d,d):
                    if score_1 > np.dot(d,d):
                        score_3 = score_2
                        score_2 = score_1
                        score_1 = np.dot(d,d)
                        img_3 = img_2
                        img_2 = img_1
                        img_1 = img2
                    else:
                        score_3 = score_2
                        score_2 = np.dot(d,d)
                        img_3 = img_2
                        img_2 = img2
                else:
                    score_3 = np.dot(d,d)
                    img_3 = img2
            print(np.dot(d,d))
        except:
            pass
        
    for i in range(0, 100):
        try:
            img2 = "/home/picture_mining/face_image_geinin/00." + str(i) + ".jpg.jpg"
            img2_rep = reps.get_rep(img2)
            d = img1_rep - img2_rep
            print(time.time() - starttime)
            print(img2)
            if score_3 > np.dot(d,d):
                if score_2 > np.dot(d,d):
                    if score_1 > np.dot(d,d):
                        score_3 = score_2
                        score_2 = score_1
                        score_1 = np.dot(d,d)
                        img_3 = img_2
                        img_2 = img_1
                        img_1 = img2
                    else:
                        score_3 = score_2
                        score_2 = np.dot(d,d)
                        img_3 = img_2
                        img_2 = img2
                else:
                    score_3 = np.dot(d,d)
                    img_3 = img2
            print(np.dot(d,d))
        except:
            pass
    print("3rd score is:" + str(score_3))
    print("3rd image is:" + img_3)
    im = Image.open(img_3)
    #画像をarrayに変換
    im_list = np.asarray(im)
    #貼り付け
    plt.imshow(im_list)
    #表示
    plt.show()
    print("2rd score is:" + str(score_2))
    print("2rd image is:" + img_2)
    im = Image.open(img_2)
    im.show()
    print("1rd score is:" + str(score_1))
    print("1rd image is:" + img_1)
    im = Image.open(img_1)
    im.show()
        
