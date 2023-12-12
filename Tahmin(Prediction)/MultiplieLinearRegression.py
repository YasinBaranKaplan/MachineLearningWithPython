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
print(veriler)
#verilerin location konumu atanır.
ulke = veriler.iloc[:,0:1].values
print(ulke)

yas=veriler.iloc[:,1:4].values
print(yas)

cinsiyet = veriler.iloc[:,-1:].values
print(cinsiyet)

#label encoding kategorik değerleri numerik değerlere çevirmek için kullanılır
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
cinsiyet[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(cinsiyet)

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

#oneHotEncoding
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

cinsiyet=ohe.fit_transform(cinsiyet).toarray()
print(cinsiyet)

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["Fr","Tr","Us"])
print(sonuc)

sonuc2 =pd.DataFrame(data=yas,index=range(22),columns=["Boy","Kilo","Yas"])
print(sonuc2)

sonuc3 = pd.DataFrame(data=cinsiyet[:,0:1],index=range(22),columns=["Cinsiyet"])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
Sc=StandardScaler()

X_train=Sc.fit_transform(x_train)
X_test=Sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

y_predict=regressor.predict(x_test)

boy = s2.iloc[:,3:4]
print(boy)

sol = s2.iloc[:,:3]
sağ =s2.iloc[:,4:]

veri = pd.concat([sol,sağ],axis=1)

x2_train,x2_test,y2_train,y2_test =train_test_split(veri,boy,test_size=0.33,random_state=0)

regressor=LinearRegression()
regressor.fit(x2_train, y2_train)

y2_predict=regressor.predict(x2_test)

#bacward ellimination
import statsmodels.api as sm
X = np.append(arr = np.ones((22,1)).astype(int),values=veri,axis=1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())

X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())




