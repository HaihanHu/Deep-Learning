#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:37:14 2020

@author: shirleyhu
"""
import numpy as np
from keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

 =============================================================================
 (x_train, y_train), (x_test, y_test) = mnist.load_data()
 
 #print('Shape of x_train: ' + str(x_train.shape))
 #print('Shape of x_test: ' + str(x_test.shape))
 #print('Shape of y_train: ' + str(y_train.shape))
 #print('Shape of y_test: ' + str(y_test.shape))
 
 x_train_vec = x_train.reshape(60000, 784) 
 x_test_vec = x_test.reshape(10000, 784)
 
 def to_one_hot(labels, dimension=10):
     results = np.zeros((len(labels),dimension))
     for i, label in enumerate(labels):
         results[i, label] = 1
     return results
 y_train_vec = to_one_hot(y_train)
 y_test_vec = to_one_hot(y_test)
 
 rand_indices = np.random.permutation(60000)
 train_indices = rand_indices[0:50000]
 valid_indices = rand_indices[50000:60000]
 x_valid_vec = x_train_vec[valid_indices,:]
 y_valid_vec = y_train_vec[valid_indices,:]
 x_train_vec = x_train_vec[train_indices,:]
 y_train_vec = y_train_vec[train_indices,:]
 
 # build the softmax classifier
 model = Sequential()
 model.add(layers.Dense(10,activation='softmax',input_shape=(784,)))
 
 model.summary()
 
 model.compile(optimizers.RMSprop(lr=0.0001),
               loss='categorical_crossentropy',
               metrics=['accuracy'])
 
 history = model.fit(x_train_vec, y_train_vec,
                     batch_size=128, epochs=50,
                     validation_data=(x_valid_vec,y_valid_vec))
 
 #plot the accuracy against epochs
 epochs = range(50)
 train_acc = history.history['accuracy']
 valid_acc = history.history['val_accuracy']
 plt.plot(epochs, train_acc, 'bo', label='Train Accuracy')
 plt.plot(epochs, valid_acc, 'r', label='Validation Accuracy')
 plt.xlabel('Epochs')
 plt.ylabel('Accuracy')
 plt.legend()
 plt.show()
 
 loss_and_acc = model.evaluate(x_test_vec, y_test_vec)
 print('loss=' +str(loss_and_acc[0]))
 print('accuracy='+str(loss_and_acc[1]))
 
 =============================================================================

# Full-connected Neural Network

d1 = 500
d2 = 500

model = models.Sequential()
model.add(layers.Dense(d1, activation='relu', input_shape=(784,)))
model.add(layers.Dense(d2, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizers.RMSprop(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train_vec, y_train_vec,
                    batch_size=128, epochs=50,
                    validation_data=(x_valid_vec,y_valid_vec))
#plot the accuracy against epochs
epochs = range(50)
train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']
plt.plot(epochs, train_acc, 'bo', label='Train Accuracy')
plt.plot(epochs, valid_acc, 'r', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss_and_acc = model.evaluate(x_test_vec, y_test_vec)
print('loss=' +str(loss_and_acc[0]))
print('accuracy='+str(loss_and_acc[1]))