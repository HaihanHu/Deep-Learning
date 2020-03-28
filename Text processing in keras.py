#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:57:32 2020

@author: shirleyhu
"""
import numpy as np
import pandas as pd
import os
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

# Load data

train_path = '/Users/shirleyhu/Downloads/aclImdb/train/'

def imdb_data_collect(inpath, outpath='/Users/shirleyhu/Desktop/', name='imdb_train.csv'):
    indices = []
    text = []
    rating = []
    i = 0 
    for file in os.listdir(inpath+'pos'):
        data = open(inpath+'pos/'+file, 'r').read()
        indices.append(i)
        text.append(data)
        rating.append('1')
        i = i + 1
    for file in os.listdir(inpath+'neg'):
        data = open(inpath+'neg/'+file, 'r').read()
        indices.append(i)
        text.append(data)
        rating.append('0')
        i = i + 1
    dataset = list(zip(indices,text,rating))
    np.random.shuffle(dataset)
    df =pd.DataFrame(data = dataset, columns=['index','text','rating'])
    df.to_csv(outpath+name, index=False, header=True)
    
    pass

imdb_data_collect(train_path)

def retrieve_data(name='/Users/shirleyhu/Desktop/imdb_train.csv', train=True):
    data = pd.read_csv(name, header=0)
    x = data['text']
    if train:
        y = data['rating']
        return x, y
    return x
[x_train, y_train] = retrieve_data()

def to_one_hot(labels, dimension=2):
     results = np.zeros((len(labels),dimension))
     for i, label in enumerate(labels):
         results[i, label] = 1
     return results
y_train = to_one_hot(y_train)

# Text to Sequence

vocabulary = 10000
tokenizer = Tokenizer(num_words=vocabulary)
tokenizer.fit_on_texts(x_train)

word_index = tokenizer.word_index
sequences_train = tokenizer.texts_to_sequences(x_train)

print(sequences_train[0])

# Align sequences

word_num = 20
x_train = preprocessing.sequence.pad_sequences(sequences_train, maxlen=word_num)

print(x_train[0])

# Word Embedding

embedding_dim = 8
model = Sequential()
model.add(Embedding(vocabulary, embedding_dim, input_length=word_num))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
model.summary()

epochs = 50
model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy', metrics=['acc'])

# Partition to training and validation
rand_indices = np.random.permutation(25000)
train_indices = rand_indices[:20000]
valid_indices = rand_indices[20000:]
x_valid_vec = x_train[valid_indices,:]
y_valid_vec = y_train[valid_indices,:]
x_train_vec = x_train[train_indices,:]
y_train_vec = y_train[train_indices,:]

history = model.fit(x_train_vec, y_train_vec, epochs=epochs,
                    batch_size=32, validation_data=(x_valid_vec, y_valid_vec))

# Plot
epochs = range(50)
train_acc = history.history['acc']
valid_acc = history.history['val_acc']
plt.plot(epochs, train_acc, 'bo', label='Train Accuracy')
plt.plot(epochs, valid_acc, 'r', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Performance on test
loss_and_acc = model.evaluate(x_test, y_test)
print('loss = '+ str(loss_and_acc[0]))
print('acc = '+ str(loss_and_acc[1]))
