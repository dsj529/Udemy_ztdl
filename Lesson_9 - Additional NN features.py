#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:05:54 2020

@author: dsj529
"""

#%%
## exercise 1 -- imdb sentiment
# a) load and shape the data
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.datasets import imdb
import keras.backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv2D, MaxPool2D
from keras.layers import LSTM, GRU
from keras.layers import Flatten, Activation
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

(X_train, y_train), (X_test, y_test) = imdb.load_data('/tmp/imdb.npz',
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
idx = imdb.get_word_index()
# create reverse-lookup index
rev_idx = {v+3:k for k, v in idx.items()}
rev_idx[0] = 'padding_char'
rev_idx[1] = 'start_char'
rev_idx[2] = 'oov_char'
rev_idx[3] = 'unknown_char'

maxlen = 100
X_train_pad = pad_sequences(X_train, maxlen=maxlen)
X_test_pad = pad_sequences(X_test, maxlen=maxlen)

max_features = max([max(x) for x in X_train_pad] + 
                   [max(x) for x in X_test_pad]) + 1

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train_pad, y_train,
          batch_size=32,
          epochs=2,
          validation_split=0.3) 
score, acc = model.evaluate(X_test_pad, y_test)
print('Test score', score)
print('Test acc', acc)

#%%
#b) retrain, using the top 20K common terms, 80 word segments (from the beginnings of the reviews)
from collections import Counter

X_train_pad = pad_sequences(X_train, maxlen=80, truncating='post')
X_test_pad = pad_sequences(X_test, maxlen=80, truncating='post')

c = Counter(X for Xs in X_train_pad for X in Xs)
c.update(X for Xs in X_test_pad for X in Xs)
common = c.most_common(20000)

K.clear_session()
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train_pad, y_train,
          batch_size=32,
          epochs=2,
          validation_split=0.3) 
score, acc = model.evaluate(X_test_pad, y_test)
print('Test score', score)
print('Test acc', acc)

#%%
## Exercise 2 -- regularization and dropout
#a) load the data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = digits.data, digits.target
y_cat = to_categorical(y, 10)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3)

def repeated_dropout_and_regularization(X_train,
                                        y_train,
                                        X_test,
                                        y_test,
                                        units=512,
                                        activation='relu',
                                        optimizer='rmsprop',
                                        do_dr=False,
                                        epochs=10,
                                        repeats=5):
    histories = []
    if do_dr:
        drop=0.3
        regularizer = 'l2'
    else:
        drop=0.0
        regularizer = None
    for repeat in range(repeats):
        K.clear_session()
        model = Sequential()
        model.add(Dense(units, 
                        input_shape=X_train.shape[1:],
                        kernel_initializer='normal',
                        kernel_regularizer=regularizer,
                        activation=activation))
        model.add(Dropout(drop))
        model.add(Dense(units, 
                        kernel_initializer='normal',
                        kernel_regularizer=regularizer,
                        activation=activation))
        model.add(Dropout(drop))
        model.add(Dense(units, 
                        kernel_initializer='normal',
                        kernel_regularizer=regularizer,
                        activation=activation))
        model.add(Dropout(drop))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        h = model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      epochs=epochs,
                      verbose=0)
        histories.append([h.history['acc'], h.history['val_acc']])
        print(repeat, end=' ')
        
    histories=np.array(histories)
    mean_acc = histories.mean(axis=0)
    std_acc = histories.std(axis=0)
    print()
    
    return mean_acc[0], std_acc[0], mean_acc[1], std_acc[1]


mean_acc, std_acc, mean_acc_val, std_acc_val = \
    repeated_dropout_and_regularization(X_train, y_train, X_test, y_test, do_dr=False)
    
mean_acc_dr, std_acc_dr, mean_acc_val_dr, std_acc_val_dr = \
    repeated_dropout_and_regularization(X_train, y_train, X_test, y_test, do_dr=True)
    
def plot_mean_std(m, s):
    plt.plot(m)
    plt.fill_between(range(len(m)), m-s, m+s, alpha=0.1)
    
plot_mean_std(mean_acc, std_acc)
plot_mean_std(mean_acc_val, std_acc_val)
plot_mean_std(mean_acc_dr, std_acc_dr)
plot_mean_std(mean_acc_val_dr, std_acc_val_dr)
plt.ylim(0.6, 1.01)
plt.title('Dropout-Regularization Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test', 'Train with DR', 'Test with DR'])
plt.show()

#%%
## Exercise 3 -- Crowdflower data for image recognition 
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale = 1./255,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             rotation_range = 20,
                             shear_range = 0.3,
                             zoom_range = 0.3,
                             horizontal_flip = True,
                             vertical_flip=True)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), data_format='channels_last'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), data_format='channels_last'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), data_format='channels_last'))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16

train_generator = datagen.flow_from_directory('./data/data/train',
                                              target_size=(128, 128),
                                              batch_size=batch_size,
                                              class_mode='binary')  

val_generator= datagen.flow_from_directory('./data/data/test',
                                           target_size=(128, 128),
                                           batch_size=batch_size,
                                           class_mode='binary')

model.fit_generator(train_generator,
                    steps_per_epoch=2500 // batch_size,
                    epochs=50,
                    validation_data=val_generator,
                    validation_steps=1000 // batch_size)