#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:13:18 2020

@author: dsj529
"""
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

## Exercise 1
df = pd.read_csv('../data/international-airline-passengers.csv')

print(df.head())
print(df.info())

df.Month = pd.to_datetime(df.Month)
df = df.set_index('Month')

df.plot()


## Exercise 2
df = pd.read_csv('../data/weight-height.csv')
df.head()
df.info()
df.plot(kind='scatter', x='Height', y='Weight')

df['GenderColor'] = df.Gender.map({'Male':'blue', 'Female':'red'})
df.plot(kind='scatter', x='Height', y='Weight', c=df.GenderColor, alpha=0.4)


## Exercise 3
fig, ax = plt.subplots()
df[df.Gender == 'Male'].hist(column='Height', ax=ax, color='blue', alpha=0.4, bins=200, range=(50, 80))
df[df.Gender == 'Female'].hist(column='Height', ax=ax, color='red', alpha=0.4, bins=200, range=(50, 80))
plt.axvline(df[df.Gender == 'Male'].Height.mean(), color='blue')
plt.axvline(df[df.Gender == 'Female'].Height.mean(), color='red')
plt.title('Height distribution')
plt.legend(['Males', 'Females'])
plt.xlabel('Height (in)')


# Exercise 4
df2 = df.pivot(columns='Gender', values='Weight')
df2.head()
df2.plot(kind='box')

## Exercise 5
df3 = pd.read_csv('../data/titanic-train.csv')
from pandas.plotting import scatter_matrix
scatter_matrix(df3, figsize=(10,10), alpha=0.2, diagonal='kde')
