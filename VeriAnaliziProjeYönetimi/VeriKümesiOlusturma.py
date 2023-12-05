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

print(list(range(22)))
sonuc = pd.DataFrame(data=ülke,index = range(22),columns=["Fr","Tr","Us"])
print(sonuc)


sonuc2= pd.DataFrame(data=Yas,index=range(22),columns=["Boy","Kilo","Yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
#print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet,index=range(22),columns=["Cinsiyet"])
print(sonuc3)

s=pd.concat([sonuc,sonuc2])
print(s)#şuan buradaki işlem dikey ekleme gibi ancak axis komutunu eklersek yan yana bir ekleme olacak

s2=pd.concat([sonuc,sonuc2],axis=1)
print(s2)

s3=pd.concat([s2,sonuc3],axis=1)
print(s3)


from sklearn.model_selection import train_test_split#split=bölmek

x_train, x_test, y_train,y_test =train_test_split(s2,sonuc3,test_size=0.33,random_state=0)




