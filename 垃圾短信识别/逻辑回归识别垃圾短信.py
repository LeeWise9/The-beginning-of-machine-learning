# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 21:08:13 2019

@author: Leo
"""

#逻辑回归识别垃圾短信

import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('SMSSpamCollection.txt',delimiter='\t',header=None)
y,X_train = df[0],df[1]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_train)

lr =linear_model.LogisticRegression()
lr.fit(X,y)

testX = vectorizer.transform(['I plane to give on this month end.',
                              'Please call our customer service representative!'])
predictions = lr.predict(testX)
print(predictions)
