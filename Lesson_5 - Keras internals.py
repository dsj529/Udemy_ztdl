#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:21:30 2020

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

import keras.backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop

## Exercise 1 -- wine
# a) load data and explore it
wine = pd.read_csv('../data/wines.csv')
print(wine.head())
print(wine.describe())
wine.hist()

sns.pairplot(wine, hue='Class', diag_kind='hist')

# b) rescale for analysis
sc = StandardScaler()
X = sc.fit_transform(wine.drop('Class', axis=1).values)
y = wine.Class.values
y_cat = pd.get_dummies(y).values

# c) Build a deep model to classify the set
K.clear_session()
model = Sequential()
model.add(Dense(5, input_shape=(13,),
                kernel_initializer='he_normal',
                activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(RMSprop(lr=0.01),
              'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y_cat, batch_size=8, epochs=20, verbose=1, validation_split=0.2)

## Exercise 2
# a) build deeper model
K.clear_session()
model = Sequential()
model.add(Dense(8, input_shape=(13,),
                kernel_initializer='he_normal',
                activation='relu'))
model.add(Dense(5, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(2, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(3, kernel_initializer='he_normal', activation='softmax'))
model.compile(RMSprop(lr=0.05),
              'categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, y_cat, batch_size=8, epochs=25, verbose=1, validation_split=0.2)              

# b) add a feature function between L1 input/l3 out
K.clear_session()
model = Sequential()
model.add(Dense(8, input_shape=(13,),
                kernel_initializer='he_normal',
                activation='relu'))
model.add(Dense(5, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(2, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(3, kernel_initializer='he_normal', activation='softmax'))
model.compile(RMSprop(lr=0.05),
              'categorical_crossentropy',
              metrics=['accuracy'])

f_in = model.layers[0].input
f_out = model.layers[2].output
feature_function = K.function([f_in], [f_out])
features = feature_function([X])[0]

for i in range(1,26):
    plt.subplot(5,5,i)
    h = model.fit(X, y_cat, batch_size=8, epochs=1, verbose=1)
    test_accuracy = model.evaluate(X, y_cat)[1]
    plt.scatter(features[:,0], features[:, 1], c=y_cat)
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.5, 4.0)
    plt.title('Epoch: {}, Test acc: {:3.1f}'.format(i, test_accuracy*100))
plt.tight_layout()

## Exercise 3 -- using the Keras functional API
from keras.layers import Input
from keras.models import Model

K.clear_session()
inputs = Input(shape=(13,))
x = Dense(8, kernel_initializer='he_normal', activation='relu')(inputs)
x = Dense(5, kernel_initializer='he_normal', activation='relu')(x)
second_to_last = Dense(2, kernel_initializer='he_normal', activation='relu')(x)
outputs = Dense(3, activation='softmax')(second_to_last)

model = Model(inputs=inputs, outputs=outputs)
model.compile(RMSprop(lr=0.05),
              'categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, y_cat, batch_size=8, epochs=25, verbose=1)

feature_function = K.function([inputs], [second_to_last])
features = feature_function([X])[0]
plt.scatter(features[:,0], features[:, 1], c=y_cat)

## Exercuse 4 -- callbacks
# a) import and instantiate callback methods
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

checkpointer = ModelCheckpoint(filepath='/tmp/udemy/weights.hdf5',
                               verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=1, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir='/tmp/udemy/tensorboard/')

#b) create and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3)

K.clear_session()
inputs = Input(shape=(13,))
x = Dense(8, kernel_initializer='he_normal', activation='relu')(inputs)
x = Dense(5, kernel_initializer='he_normal', activation='relu')(x)
second_to_last = Dense(2, kernel_initializer='he_normal', activation='relu')(x)
outputs = Dense(3, activation='softmax')(second_to_last)

model = Model(inputs=inputs, outputs=outputs)
model.compile(RMSprop(lr=0.05),
              'categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=8, epochs=25, verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[checkpointer, earlystopper, tensorboard])
