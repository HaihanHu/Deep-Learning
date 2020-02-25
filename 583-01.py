#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:02:56 2020

@author: shirleyhu
"""

import numpy as np
import numpy.linalg as LA

a = np.random.rand(6,7)
b = np.random.rand(7,8)
c = np.dot(a,b)
print(c.shape)
print(c.shape[1])


def compute_norm():
    mat = np.matrix([[-1,0,2],[4,-5,3]])
    inv_mat = np.linalg.inv(mat)
    print (inv_mat)
    
def vector_norm():
    a = np.array([3,-10,9,0,-2])
    print (a)
    print (LA.norm(a,np.inf)) #无穷norm:max绝对值
    print (LA.norm(a,-np.inf))
    print (LA.norm(a,1)) #norm1:绝对值之和
    print (LA.norm(a,2)) #norm2:平方和开方
    
#def matrix_norm():
#    a = np.matrix([[-1,0,2],[4,-5,3]])
#    b = np.array([5,6,7])
#    b_t = np.transpose(b)
#    b_new = np.dot(b_t,b) #b_new矩阵为b^t * b
#    x = np.linalg.eigvals(b_new) #求b_new矩阵的特征值
#    print x
#    print LA.norm(b,1) #列范数
#    print LA.norm(b,2) #谱范数,为x里最大值开平方
#    print LA.norm(b,np.inf) #无穷范数，行范数
#    print LA.norm(b,"fro") #F范数

vector_norm()

#b = np.identity(5, dtype = float) 
#b_tr = np.trace(b)
##eigenvalue,featurevector=np.linalg.eig(b)
#b_Fnorm = LA.norm(b,"fro")

x = np.array([3,-10,9,0,-2])
a1 = np.array([0,9,-3,-2,1])
xx = np.dot(x,a1.T)

md = LA.norm(x-a1,1) #norm1:绝对值之和

mat = np.matrix([[-1,0,2],[4,-5,3]])
xx_A = np.dot(mat,mat.T)
A_tr = np.trace(xx_A)

a9 = np.matrix([[-1,0,2],[4,-5,3]])
b9 = np.array([5,6,7])
new_9 = np.dot(a9,b9)
Ab = LA.norm(new_9,2)