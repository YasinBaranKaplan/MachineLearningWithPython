# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 20:15:23 2023

@author: kapla
"""

#imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#code 
#data load

veriler = pd.read_csv('veriler.csv')
print(veriler)

boy = veriler[['boy']]
boykilo = veriler[['boy','kilo']]

print(boykilo)

class insan:
    boy = 180
    def kosmak(b):
        return b+10

ali = insan()
print(ali.boy)

l=[1,3,4];