# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:03:58 2023

@author: kapla
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#eksik veriler
eksikveriler = pd.read_csv('eksikveriler.csv')
veriler = pd.read_csv('veriler.csv')

print(eksikveriler)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
Yas = eksikveriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])

print(Yas)

ülke = veriler.iloc[:,0:1].values
print(ülke)

#label encoding kategorik değerleri numerik değerlere çevirmek için kullanılır
from sklearn import preprocessing
#label encoding = le
le=preprocessing.LabelEncoder()
ülke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ülke)

#oneHotEncoding
ohe=preprocessing.OneHotEncoder()
ülke=ohe.fit_transform(ülke).toarray()
print(ülke)













