#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:06:08 2020

@author: dsj529
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

## Exercise 1 -- UCI Pima Indian dataset
# a) load data 
pima = pd.read_csv('../data/diabetes.csv')
print(pima.head())
print(pima.describe()) 

# b) plot histograms of the features
# for i, col in enumerate(pima.columns):
#     plt.subplot(1, len(pima.columns), i+1)
#     pima[col].plot(kind='hist', title=col, sharey=True)
#     plt.xlabel(col)

_ = pima.hist(figsize=(12,10))    
# c) explore correlation of feature columns
sns.heatmap(pima.corr(), annot=True)
sns.pairplot(pima, hue='Outcome', diag_kind='hist')

# d) rescale the columns
sc = StandardScaler()
X = sc.fit_transform(pima.drop('Outcome', axis=1))
y = pima['Outcome']
y_cat = to_categorical(y)

## Exercise 2 -- build an ANN over the Pima data
# a) create test/train subsets
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

# b) build the ANN model
model = Sequential()
model.add(Dense(32, input_shape=(8,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(Adam(lr=0.05),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# c) evaluate the results
model.fit(X_train, y_train, epochs=50, verbose=2, validation_split=0.1)
y_pred = model.predict(X_test)
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

print(accuracy_score(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class))

## Exercise 3 -- compare with non-NN algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

for mod, clf in [('Random Forest', RandomForestClassifier()), 
                 ('SVM', SVC()),
                 ('Naive Bayes', GaussianNB())]:
    clf.fit(X_train, y_train[:, 1])
    y_pred = clf.predict(X_test)
    print('='*80)
    print(mod)
    print('Accuracy: {:.3f}'.format(accuracy_score(y_test_class, y_pred)))
    print('Confusion matrix: ')
    print(confusion_matrix(y_test_class, y_pred))