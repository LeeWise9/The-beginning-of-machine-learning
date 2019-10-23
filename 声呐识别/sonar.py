# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:14:01 2019

@author: Leo
"""

#声呐分类
#Dropout
#maxnorm

import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder #将数据集中文本转为数字
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 7
np.random.seed(seed)

dataframe = read_csv('sonar.csv',header=None)
dataset = dataframe.values

X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y =encoder.transform(Y)

def create_baseline():
    model = Sequential()
    model.add(Dropout(0.2,input_shape=(60,)))
    model.add(Dense(60,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dense(30,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    
    sgd = SGD(lr=0.01,momentum = 0.8, decay = 0.0, nesterov = False)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

estimators = [] #使用Pipeline方式
estimators.append(('standardize', StandardScaler())) #标准化：均值为0，方差为1
estimators.append(('mlp',KerasClassifier(build_fn=create_baseline,epochs=300,batch_size=16,verbose=1))) #传入网络结构
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed) #交叉验证
results = cross_val_score(pipeline,X,encoded_Y,cv=kfold)
print('Average Accuracy: %.2f%% (%.2f%%)' % (results.mean()*100,results.std()*100))

