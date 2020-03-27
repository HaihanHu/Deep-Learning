#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:58:02 2020

@author: shirleyhu
"""

from keras.datasets import imdb
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import regularizers

(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)
#train_data[0]
max(max(sequence)for sequence in train_data)#9999

# decode reviews back to english words
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown"
#the indices were offset by 3
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

# vectorize sequences
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# vectorized lables
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# build model
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_train_new, x_valid, y_train_new, y_valid = train_test_split(x_train, y_train, test_size=0.33)
x_valid = x_train[:10000]
x_train_new = x_train[10000:]
y_valid = y_train[:10000]
y_train_new = y_train[10000:]

#history = model.fit(x_train_new,
#                    y_train_new,
#                    epochs=30,
#                    batch_size=512,
#                    validation_data=(x_valid, y_valid))
#
## plot accuracy by epochs
#acc_values = history.history['accuracy']
#val_acc_values = history.history['val_accuracy']
#epochs = range(1, 31)
#plt.plot(epochs, acc_values, 'bo', label='Training acc')
#plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#
#plt.show() #overfitting 

# add weight regularization and dropout
new_model = models.Sequential()
new_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(10000,)))
new_model.add(layers.Dropout(0.5))
new_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
new_model.add(layers.Dropout(0.5))
new_model.add(layers.Dense(1, activation = 'sigmoid'))

new_model.compile(optimizer=optimizers.RMSprop(lr=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = new_model.fit(x_train_new,
                    y_train_new,
                    epochs=30,
                    batch_size=512,
                    validation_data=(x_valid, y_valid))

# plot accuracy by epochs
acc_values = history.history['accuracy']
val_acc_values = history.history['val_accuracy']
epochs = range(1, 31)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

