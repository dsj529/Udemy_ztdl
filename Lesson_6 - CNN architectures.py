#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:25:39 2020

@author: dsj529
"""

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from keras.datasets import mnist
import keras.backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D
from keras.layers import Flatten, Activation
from keras.optimizers import Adam, RMSprop

## Exercise 1: MNIST classifiication
# a) load and shape data
(X_train, y_train), (X_test, y_test) = mnist.load_data('/tmp/mnist.npz')

# img = sp.misc.ascent()
# img_tensor = img.reshape(1, 512, 512, 1)
# X_train.shape
# plt.imshow(X_train[0], cmap='gray')

X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# b) create a model
K.clear_session()

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train_cat, batch_size=128, epochs=2,
          verbose=1, validation_split=0.3)
model.evaluate(X_test, y_test_cat)

## Exercise 2: CIFAR-10
# a) load and shape the data
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# X_train.shape
# plt.imshow(X_train[0])

X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

#b) build model
K.clear_session()

model = Sequential()
model.add(Conv2D(32, (3,3), 
                 padding='same',
                 input_shape=(32,32,3),
                 activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train_cat, 
          batch_size=32, epochs=2,
          verbose=1, 
          validation_data=(X_test, y_test_cat),
          shuffle=True)