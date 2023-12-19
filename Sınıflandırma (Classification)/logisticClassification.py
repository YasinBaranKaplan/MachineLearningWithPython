# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:03:58 2023

@author: kapla
"""
#1.Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values#bağımsız değişkenler
y=veriler.iloc[:,4:]#bağımlı değişken


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)


from sklearn.linear_model import LogisticRegression
logr=LogisticRegression()
logr.fit(x_train, y_train)

y_predict = logr.predict(x_test)


