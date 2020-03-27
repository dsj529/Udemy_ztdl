#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:45:14 2020

@author: dsj529
"""

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv('../data/banknotes.csv')
X = scale(df.drop('class', axis=1).values)
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

## compare learning rates on sigmoid
dflist = []
learning_rates = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7]

for lr in learning_rates:
    K.clear_session()
    model = Sequential()
    model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=lr),
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)
    dflist.append(pd.DataFrame(h.history, index=h.epoch))
    
historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([learning_rates, metrics_reported],
                                 names=['learning_rate', 'metric'])
historydf.columns = idx

ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title('Loss')

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title('Accuracy')
plt.xlabel('Epochs')

plt.tight_layout()

## compare optimizers
dflist = []
names = ['SGD', 'SGD momentum', 'SGD nesterov', 'Adam', 'Adagrad', 'RMSprop']
optimizers = ['SGD(lr=0.01)',
              'SGD(lr=0.01, momentum=0.3)',
              'SGD(lr=0.01, momentum=0.3, nesterov=True)',
              'Adam(lr=0.01)',
              'Adagrad(lr=0.01)',
              'RMSprop(lr=0.01)']

for opt in optimizers:
    K.clear_session()
    model = Sequential()
    model.add(Dense(1, input_shape=(4,), activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=eval(opt),
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=0)
    dflist.append(pd.DataFrame(h.history, index=h.epoch))
    
historydf = pd.concat(dflist, axis=1)
metrics_reported = dflist[0].columns
idx = pd.MultiIndex.from_product([names, metrics_reported],
                                 names=['optimizers', 'metric'])
historydf.columns = idx

ax = plt.subplot(211)
historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title('Loss')

ax = plt.subplot(212)
historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
plt.title('Accuracy')
plt.xlabel('Epochs')

plt.tight_layout()

## Visualize inner layer activity
K.clear_session()

model = Sequential()
model.add(Dense(2, input_shape=(4,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])
h = model.fit(X_train, y_train, batch_size=16, epochs=25, verbose=1, validation_split=0.2)
result=model.evaluate(X_test, y_test)

# print(result)

f_in = model.layers[0].input
f_out = model.layers[0].output
feature_function = K.function([f_in], [f_out])
features = feature_function([X_test])[0]
plt.scatter(features[:,0], features[:, 1], c=y_test, cmap='coolwarm')

K.clear_session()

model = Sequential()
model.add(Dense(3, input_shape=(4,), activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])

# print(result)

f_in = model.layers[0].input
f_out = model.layers[1].output
feature_function = K.function([f_in], [f_out])
features = feature_function([X_test])[0]
for i in range(1,26):
    plt.subplot(5,5,i)
    h = model.fit(X_train, y_train, batch_size=16, epochs=1, verbose=1)
    test_accuracy = model.evaluate(X_test, y_test)[1]
    plt.scatter(features[:,0], features[:, 1], c=y_test, cmap='coolwarm')
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.5, 4.0)
    plt.title('Epoch: {}, Test acc: {:3.1f}'.format(i, test_accuracy*100))
plt.tight_layout()