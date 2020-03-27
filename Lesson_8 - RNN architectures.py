#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:08:42 2020

@author: dsj529
"""
#%%
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from keras.datasets import mnist
import keras.backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPool2D
from keras.layers import LSTM, GRU
from keras.layers import Flatten, Activation
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler

## Exercise 1 -- Cansim data
# a) read and shape the data
df = pd.read_csv('../data/cansim-0800020-eng-6674700030567901031.csv',
                 skiprows=6, skipfooter=9, engine='python')
df.head()
from pandas.tseries.offsets import MonthEnd

df['Adjustments'] = pd.to_datetime(df['Adjustments']) + MonthEnd(1)
df = df.set_index('Adjustments')
df.plot()

split_date = pd.Timestamp('01-01-2011')

train = df.loc[:split_date, ['Unadjusted']]
test = df.loc[split_date:, ['Unadjusted']]

ax=train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])

#%%
sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)
X_train = train_sc[:-1]
y_train = train_sc[1:]
X_test = test_sc[:-1]
y_test = test_sc[1:]

# convert to tensor batches for the LSTM-NN
X_train_t = X_train[:, None]
X_test_t = X_test[:, None]

#%%
# b) build (1, 1) LSTM RNN
K.clear_session()

early_stop = EarlyStopping(monitor='loss', min_delta=0,
                             patience=1, verbose=1, mode='auto')
model = Sequential()
model.add(LSTM(6, input_shape=(1,1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

model.fit(X_train_t, y_train, epochs=100,
          batch_size=1, verbose=1,
          callbacks=[early_stop])

y_pred = model.predict(X_test_t)
plt.plot(y_test)
plt.plot(y_pred)
plt.title('(1,1) LSTM')

#%%
# C) build (1, 1) GRU RNN
K.clear_session()

early_stop = EarlyStopping(monitor='loss', min_delta=0,
                             patience=1, verbose=1, mode='auto')
model = Sequential()
model.add(GRU(12, input_shape=(1,1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

model.fit(X_train_t, y_train, epochs=100,
          batch_size=1, verbose=1,
          callbacks=[early_stop])

y_pred = model.predict(X_test_t)
plt.plot(y_test)
plt.plot(y_pred)
plt.title('(1,1) GRU')

#%%
# d) convert to (1, 12)  LSTM Model
train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)

for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['Scaled'].shift(s)
    
X_train = train_sc_df.dropna().drop('Scaled', axis=1).values
y_train = train_sc_df.dropna()[['Scaled']].values
X_test = test_sc_df.dropna().drop('Scaled', axis=1).values
y_test = test_sc_df.dropna()[['Scaled']].values

X_train_t = X_train.reshape(X_train.shape[0], 1, 12)
X_test_t = X_test.reshape(X_test.shape[0], 1, 12)

K.clear_session()
model = Sequential()
model.add(LSTM(6, input_shape=(1,12)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train_t, y_train, epochs=100,
          batch_size=1, verbose=1,
          callbacks=[early_stop])

y_pred = model.predict(X_test_t)
plt.plot(y_test)
plt.plot(y_pred)
plt.title('(1,12) LSTM')
model.evaluate(X_test_t, y_test)

#%%
# e) convert to (1, 12)  GRU Model
K.clear_session()
model = Sequential()
model.add(GRU(12, input_shape=(1,12)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train_t, y_train, epochs=100,
          batch_size=1, verbose=1,
          callbacks=[early_stop])

y_pred = model.predict(X_test_t)
plt.plot(y_test)
plt.plot(y_pred)
plt.title('(1,12) GRU')
model.evaluate(X_test_t, y_test)

#%%
# f) convert to (12, 1) LSTM model
X_train_t = X_train.reshape(X_train.shape[0], 12, 1)
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)

K.clear_session()
model = Sequential()
model.add(LSTM(6, input_shape=(12,1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train_t, y_train, epochs=100,
          batch_size=1, verbose=1,
          callbacks=[early_stop])

y_pred = model.predict(X_test_t)
plt.plot(y_test)
plt.plot(y_pred)
plt.title('(12,1) LSTM')
model.evaluate(X_test_t, y_test)

#%%
# g) convert to (12, 1) GRU model
X_train_t = X_train.reshape(X_train.shape[0], 12, 1)
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)

K.clear_session()
model = Sequential()
model.add(LSTM(6, input_shape=(12,1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train_t, y_train, epochs=100,
          batch_size=1, verbose=1,
          callbacks=[early_stop])

y_pred = model.predict(X_test_t)
plt.plot(y_test)
plt.plot(y_pred)
plt.title('(12,1) GRU')
model.evaluate(X_test_t, y_test)

#%%
## Exercise 2 - MNIST on RNN
# a) load and shape data for (784, 1) LSTM/GRU
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data('/tmp/mnist.npz')

X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0
X_train = X_train.reshape(-1, 784, 1)
X_test = X_test.reshape(-1, 784, 1)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

#%%
# b) build, train and test on (784,1) LSTM/GRU

X_train = X_train.reshape(-1, 784, 1)
X_test = X_test.reshape(-1, 784, 1)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


K.clear_session()
model = Sequential()
model.add(LSTM(6, input_shape=(784,1)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', 
              metrics=['categorical_accuracy', 'kullback_leibler_divergence'])

model.fit(X_train, y_train_cat, epochs=100,
          batch_size=128, verbose=0,
          callbacks=[early_stop])

print('Long LSTM')
print(model.evaluate(X_test, y_test_cat))

K.clear_session()
model = Sequential()
model.add(GRU(6, input_shape=(784,1)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', 
              metrics=['categorical_accuracy', 'kullback_leibler_divergence'])

model.fit(X_train, y_train_cat, epochs=100,
          batch_size=128, verbose=0,
          callbacks=[early_stop])

print('Long GRU')
print(model.evaluate(X_test, y_test_cat))

#%%
# c) build, train and test on (1,784) LSTM/GRU
X_train = X_train.reshape(-1, 1, 784)
X_test = X_test.reshape(-1, 1, 784)

K.clear_session()
model = Sequential()
model.add(LSTM(6, input_shape=(1,784)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', 
              metrics=['categorical_accuracy', 'kullback_leibler_divergence'])

model.fit(X_train, y_train_cat, epochs=100,
          batch_size=128, verbose=0,
          callbacks=[early_stop])

print('Wide LSTM')
print(model.evaluate(X_test, y_test_cat))

K.clear_session()
model = Sequential()
model.add(GRU(6, input_shape=(1,784)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', 
              metrics=['categorical_accuracy', 'kullback_leibler_divergence'])

model.fit(X_train, y_train_cat, epochs=100,
          batch_size=128, verbose=0,
          callbacks=[early_stop])

print('Wide GRU')
print(model.evaluate(X_test, y_test_cat))
