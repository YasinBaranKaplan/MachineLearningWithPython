# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:03:58 2023

@author: kapla
"""
#1.Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("satislar.csv")
print(veriler)

aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]

#print(aylar)
#print(satislar)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)