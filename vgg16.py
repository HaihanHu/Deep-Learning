#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:29:07 2020

@author: shirleyhu
"""

import os
import pandas as pd
import random
from shutil import copy2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16


#split dataset to train, validation and test
datadir_normal = '/Users/shirleyhu/Downloads/dogs-vs-cats/train'
all_data = os.listdir(datadir_normal)
#print('number of all data:'+str(len(all_data)))
random.shuffle(list(range(len(all_data))))
num = 0
train_dir = '/Users/shirleyhu/Documents/dogs-vs-cats/train/train/'
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
valid_dir = '/Users/shirleyhu/Documents/dogs-vs-cats/validation/validation/'
if not os.path.exists(valid_dir):
    os.mkdir(valid_dir)
test_dir = '/Users/shirleyhu/Documents/dogs-vs-cats/test/test/'
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
for i in list(range(len(all_data))):
    fileName = os.path.join(datadir_normal,all_data[i])
    if num < 2000:
        copy2(fileName, train_dir)
    elif num> 2000 and num< 3000:
        copy2(fileName, valid_dir)
    elif num> 3000 and num< 4000:
        copy2(fileName, test_dir)
    num += 1


#train_dir = '/Users/shirleyhu/Documents/dogs-vs-cats/train/'
#valid_dir = '/Users/shirleyhu/Documents/dogs-vs-cats/validation/'
#test_dir = '/Users/shirleyhu/Documents/dogs-vs-cats/test/'

# binary category
filenames = os.listdir('/Users/shirleyhu/Documents/dogs-vs-cats/train/train')
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(str(1))
    else:
        categories.append(str(0))

train_df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

#filenames = os.listdir('/Users/shirleyhu/Documents/dogs-vs-cats/validation/validation')
#filenames = os.listdir('/Users/shirleyhu/Documents/dogs-vs-cats/test/test')    


# images rescaled by 1./255
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)
train_generator = train_datagen.flow_from_dataframe(
        train_df,
        '/Users/shirleyhu/Documents/dogs-vs-cats/train/train/',
        x_col = 'filename',
        y_col = 'category',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_dataframe(
        valid_df,
        '/Users/shirleyhu/Documents/dogs-vs-cats/validation/validation/',
        x_col = 'filename',
        y_col = 'category',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# build model
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))
conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
conv_base.trainable=False
model.summary()

trainable_layer_names = ['block5_conv1','block5_conv2','block5_conv3','block5_pool']
conv_base.trainable = True

for layer in conv_base.layers:
    if layer.name in trainable_layer_names:
        layer.trainable = True
    else:
        layer.trainable = False
model.summary()

model.compile(optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=10,
        validation_data=valid_generator,
        validation_steps=50)

# evaluate the model on the test
test_generator = test_datagen.flow_from_dataframe(
        test_df,
        '/Users/shirleyhu/Documents/dogs-vs-cats/test/test/',
        x_col = 'filename',
        y_col = 'category',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator,steps=50)
print('test accuracy:', test_acc)