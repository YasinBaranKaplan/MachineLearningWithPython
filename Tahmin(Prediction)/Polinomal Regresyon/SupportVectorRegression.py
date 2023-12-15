# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:51:33 2023

@author: kapla
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#veri yükleme
veriler = pd.read_csv("maaslar.csv")

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
X=x.values
Y=y.values

#Liear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

linearpredict= lin_reg.predict(x)

plt.scatter(X,Y,color='red')
plt.plot(X,linearpredict,color='blue')
plt.show()

#polynomial regression  // nonlineer model oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)#
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

#degree arttıkça tahmin gerçek sonuçlara yaklaşır.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)#
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)


from sklearn.svm import SVR
svr_reg=SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli, y_olcekli,color = 'red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')


























