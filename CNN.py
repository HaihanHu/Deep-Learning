#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:50:39 2020

@author: shirleyhu
"""

import numpy as np
from keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert 60000*28*28 to 60000*28*28*1 tensor 
x_train_vec = x_train.reshape((60000, 28, 28, 1))/255.0 
x_test_vec = x_test.reshape((10000, 28, 28, 1))/255.0

# convert labels(0-9) to 10-dim vectors
def to_one_hot(labels, dimension=10):
     results = np.zeros((len(labels),dimension))
     for i, label in enumerate(labels):
         results[i, label] = 1
     return results
y_train_vec = to_one_hot(y_train)
y_test_vec = to_one_hot(y_test)

# partition to training and validation
rand_indices = np.random.permutation(60000)
train_indices = rand_indices[0:50000]
valid_indices = rand_indices[50000:60000]
x_valid_vec = x_train_vec[valid_indices,:]
y_valid_vec = y_train_vec[valid_indices,:]
x_train_vec = x_train_vec[train_indices,:]
y_train_vec = y_train_vec[train_indices,:]

# build cnn
model = models.Sequential()
model.add(layers.Conv2D(10,#filter number
                        (5, 5),#filter shape
                        activation='relu',
                        input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2))) #pooling size
model.add(layers.Conv2D(20,(5,5),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# optimization
model.compile(optimizers.RMSprop(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train_vec, y_train_vec,
                    batch_size=128, epochs=50,
                    validation_data=(x_valid_vec,y_valid_vec))

# plot the accuracy against epochs
epochs = range(50)
train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']
plt.plot(epochs, train_acc, 'bo', label='Train Accuracy')
plt.plot(epochs, valid_acc, 'r', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# evaluation on the test set
loss_and_acc = model.evaluate(x_test_vec, y_test_vec)
print('loss=' +str(loss_and_acc[0]))
print('accuracy='+str(loss_and_acc[1]))