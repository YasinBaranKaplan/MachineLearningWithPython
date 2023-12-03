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