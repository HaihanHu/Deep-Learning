#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:02:59 2020

@author: shirleyhu
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(30)

c_images = np.random.randn(700,2) + np.array([0,-3])
m_images = np.random.randn(700,2) + np.array([3,3])
d_images = np.random.randn(700,2) + np.array([-3,3])

feature_set = np.vstack([c_images, m_images, d_images])
labels = np.array([0]*700+[1]*700+[2]*700)
one_hot_labels = np.zeros((2100,3))
for i in range(2100):
    one_hot_labels[i,labels[i]]= 1
    
plt.scatter(feature_set[:,0],feature_set[:,1],c=labels)
plt.show()

def softmax(A):
    expA= np.exp(A)
    return expA/expA.sum(axis=1,keepdims=True)

#nums = np.array([4,5,6])
#print(softmax(nums))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):#对sigmoid函数求导：(u/v)'=(u'*v-u*v')/v²
    return sigmoid(x) * (1-sigmoid(x))

instances = feature_set.shape[0]
attributes = feature_set.shape[1]
hidden_nodes = 4
output_labels = 3

wh = np.random.rand(attributes,hidden_nodes)#(x,y),4
bh = np.random.rand(hidden_nodes) #4

wo = np.random.rand(hidden_nodes, output_labels) # 4-by-3
bo = np.random.rand(output_labels)
lr = 10e-4

error_cost = []

for epoch in range(50000):
    #feedforward
    
    #phase1
    zh = np.dot(feature_set, wh)+ bh #2100by4
    ah = sigmoid(zh)
    #phase2
    zo = np.dot(ah, wo)+ bo #2100by3
    ao = softmax(zo)
    
    #back propagation
    #phase1
    dcost_dzo = ao - one_hot_labels
    dzo_dwo = ah
    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo) #dot(2100by4.T,2100by3)=4by3
    dcost_bo = dcost_dzo #2100by3
    #phase2
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo, dzo_dah.T) #4by3
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T,dah_dzh * dcost_dah) #2by4
    
    dcost_bh = dcost_dah * dah_dzh
    
    ##updat weights
    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0) # 按列sum
    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)
    
    if epoch % 100 == 0:
        loss = np.sum(-one_hot_labels * np.log(ao))
        error_cost.append(loss)
