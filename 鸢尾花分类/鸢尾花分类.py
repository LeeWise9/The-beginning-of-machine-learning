# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:29:25 2019

@author: Leo
"""

#鸢尾花分类
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold #将数据集分成K个小数据集（每次选一个作为测试集）
from sklearn.preprocessing import LabelEncoder #将数据集中文本转为数字
from keras.models import model_from_json #保存模型

# reproducibility 可重复性
seed = 13
np.random.seed(seed)

# load data
df = pd.read_csv('iris.csv')
X = df.values[:,0:4].astype(float)
Y = df.values[:,4]

encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot  = np_utils.to_categorical(Y_encoded)

# define a network
def baseline_model():
    model = Sequential()
    model.add(Dense(7,input_dim=4,activation='tanh'))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model,epochs=20,batch_size=1,verbose=1)

# evaluate
kfold = KFold(n_splits=10,shuffle=True,random_state=seed) #十次交叉验证
result = cross_val_score(estimator,X,Y_onehot,cv=kfold)
print('Accuracy of cross validation, mean %.2f, std %.2f' % (result.mean(),result.std()))

# save model
estimator.fit(X,Y_onehot)
model_json = estimator.model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)

estimator.model.save_weights('model.h5')
print('saved model to disk')

# load model and use it for prediction
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
print('loaded model from disk')

predicted = loaded_model.predict(X)
print('predicted probability:'+str(predicted))

predicted_label = loaded_model.predict_classes(X)
print('predicted label:'+str(predicted_label))

