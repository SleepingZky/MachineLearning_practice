#coding=utf-8
import pandas as pd
import numpy as numpy
import cv2
import random
import time

from sklearn.cross_decomposition import train_test_split
from sklearn.metrics import accuracy_score


def get_hog_features(trainset):
    return 

def Predict(testset, trainset, train_labels):
    precidt=[]
    count = 0

    for test_vec in testset:
        # 输出当前运行的测试用例坐标， 用于测试
        print(count)
        count += 1

        knn_list=[]
        max_index = -1
        max_dist = 0

        
    


k=10

if __name__ == "__main__":
    print('Start read data')
    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values

    imgs = data[:,1:]
    labels = data[:,0]

    features = get_hog_features(imgs)

    #2/3作为训练集，1/3作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print('read data cost:',time_2-time_1,'seconds')

    print('Start training')
    print('knn do not need to train')

    print('Start predicting')
    test_predict = Predict(test_features, train_features, train_labels)
    time_3 = time.time()
    print('predicting cost:',time_3-time_2,'seconds')
    score = accuracy_score(test_labels, test_predict)
    print('Acc:',score)