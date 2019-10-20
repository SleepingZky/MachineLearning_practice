# encoding-utf-8
# @Author:  zky
# @Data:    2019-10-20
# @Email:   zky_archi@gmail.com

import pandas as pd
import numpy as np
import cv2
import random
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 为什么不并行的实现版本准确率高很多？？
# 这个版本是89%
# 不并行的版本是98%

class Perceptron(object):
    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 300000

    def predict_(self,x):
        wx = np.matmul(x,self.w)
        # np.sign 判断元素的符号 >0:1 <0:-1
        return np.sign(wx)

    def train(self, features, labels):
        self.w = np.array([0.0]*(features.shape[1]+1),dtype=np.float)

        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = np.random.randint(low=0,high=(labels.shape[0]))
            x = np.array(features[index])
            x = np.append(x,1.0)
            y = np.array(2*labels[index]-1)
            x = np.reshape(x,(1,-1))
            wx = np.matmul(x,self.w)

            if wx*y>0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue
            delta = self.learning_step*(y*x)
            self.w += np.reshape(delta,self.w.shape)

    def predict(self, feature):
        ones = np.ones((feature.shape[0],1))
        x = np.column_stack((feature,ones))
        return self.predict_(x)

if __name__ == "__main__":

    print('Start read data')
    time_1 = time.time()
    raw_data = pd.read_csv('D:\统计机器学习\lihang_book_algorithm\data\\train_binary.csv', header=0)
    data = raw_data.values

    imgs = np.array(data[:,1:])
    labels = np.array(data[:,0])

    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print('Start training')
    p = Perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    print('Start predicting')
    test_predict = p.predict(test_features)
    print(test_predict)
    time_4 = time.time()
    print('predicting cost ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy socre is ", score)
