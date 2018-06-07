# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     pca
   
   Description :
   Author :       付学明
   date：          2018/5/24
-------------------------------------------------
   Change Activity:
                   2018/5/24:
-------------------------------------------------
__author__ = '付学明'
"""
import numpy as np
from matplotlib import pyplot as plt

X = np.load('PicData.npy',)

picnum=10

pic1=X[picnum,:98304]
m=int(X[picnum,98304])
n=int(X[picnum,98305])
pic1=pic1.reshape((m,n))
plt.imshow(pic1,cmap='gray')
plt.show()

data = X[:,:98304]


def Normalize(data):
    m,n = np.shape(data)
    mu = np.zeros((1,n))
    sigma = np.zeros((1,n))

    mu = np.mean(data,axis=0)  #按照列求均值 即个样本的均值
    sigma = np.std(X,axis=0)
    for j in range(n):
        data[:,j]=(data[:,j]-mu[j])/sigma[j]
    return data,mu,sigma

data_copy = data.copy()
X_norm, mu, sigma = Normalize(data.T)
'''
pic1=X_norm[:,picnum]
m=int(X[picnum,98304])
n=int(X[picnum,98305])
pic1=pic1.reshape((m,n))
plt.imshow(pic1,cmap='gray')
plt.show()
'''
Sigma = np.dot(np.transpose(X_norm[:,:100]), X_norm[:,:100]) / m

U, S, V = np.linalg.svd(Sigma)

threshold = 0.95
for k in range(n):
    ErrorRatio=np.sum(S[:k])/np.sum(S)
    if ErrorRatio >= threshold:
        break
#print(ErrorRatio)
print(k)

K=k
Z= np.zeros((X_norm.shape[0],K))
U_reduce = U[:,0:K]
Z = np.dot(X_norm[:,:100],U_reduce)
print(Z.shape)
U_reduce=U[:,0:K]
X_rec = np.dot(Z,np.transpose(U_reduce))

X_rec = X_rec.T

pic1=X_rec[picnum,:]
m=int(X[picnum,98304])
n=int(X[picnum,98305])
pic1=pic1.reshape((m,n))
plt.imshow(pic1,cmap='gray')
plt.show()
