# -*- coding: utf-8 -*-
"""
@author: Zhihai Wang
"""

from scipy.io import loadmat
import os 
import numpy as np
from sklearn import preprocessing

def logre(x,w):
    return 1 / ( 1 + np.exp( np.matmul(x,w) ))



# read data 

data_train_img = loadmat('hw3_lr/train_imgs.mat')
data_train_label = loadmat('hw3_lr/train_labels.mat')
data_test_img = loadmat('hw3_lr/test_imgs.mat')
data_test_label = loadmat('hw3_lr/test_labels.mat')

data_train_img = data_train_img['train_img']
data_train_label = data_train_label['train_label']
data_test_img = data_test_img['test_img']
data_test_label = data_test_label['test_label']

# data preprocess

data_train_img = data_train_img.toarray()
data_test_img = data_test_img.toarray()

data_train_label = data_train_label.toarray()
data_test_label = data_test_label.toarray()

data_train_label = data_train_label - 1
data_test_label = data_test_label - 1


(len_r_train , len_c) = data_train_img.shape
len_r_test = data_test_img.shape[0]

col_train = np.ones(len_r_train)
col_test = np.ones(len_r_test)

data_train_img = np.column_stack((data_train_img , col_train))
data_test_img = np.column_stack((data_test_img , col_test))
len_c += 1 

# normlize minmax

min_max_scaler = preprocessing.MinMaxScaler()

data_train_img_trans = data_train_img.transpose()
data_train_img_norm = min_max_scaler.fit_transform(data_train_img_trans)

data_test_img_trans = data_test_img.transpose()
data_test_img_norm = min_max_scaler.fit_transform(data_test_img_trans)

data_xtrain = data_train_img_norm.transpose()
data_xtest = data_test_img_norm.transpose()
data_ytrain = data_train_label.transpose()
data_ytest = data_test_label.transpose()

# parameters 

w = np.zeros((len_c,1))
lr = 0.01
epoch = 100 

confusion_marix = [0,0,0,0]
# train 

for i in range(epoch):

    # test 

    error = 0.0
    accuracy = 0.0

    pred = logre(data_xtest , w)    
    for j in range(len_r_test):
        if pred[j] > 0.5: 
            if data_ytest[j] == 1:
                error += 1
        else:
            if data_ytest[j] == 0:
                error += 1 
    

    accuracy = 1 - error / len_r_test
    print(accuracy)
    
    # update
    h = 1 / (1 + np.exp( -1 * np.matmul(data_xtrain , w ))) - data_ytrain
    grad = np.matmul(data_xtrain.transpose() , h)
    w = w - lr * grad
  
pred = logre(data_xtest , w)    
for j in range(len_r_test):
    if pred[j] > 0.5 and data_ytest[j] == 0:
        confusion_marix[2] += 1  # TN
        if data_ytest[j] == 1:
            error += 1
    elif pred[j] > 0.5 and data_ytest[j] == 1:
        confusion_marix[3] += 1 # FN

    elif pred[j] <= 0.5 and data_ytest[j] == 1:
        confusion_marix[0] += 1 # TP 
    elif pred[j] <= 0.5 and data_ytest[j] == 0: 
        confusion_marix[1] += 1 # FP

print("The confusion matrix:")
print(confusion_marix)
print("Precision:")
Precision = confusion_marix[0]/(confusion_marix[0]+confusion_marix[1])
print(Precision)
print("Recall:")
Recall = confusion_marix[0]/(confusion_marix[0]+confusion_marix[3])
print(Recall)
print("F1:")
print(2/(1/Precision+1/Recall))
