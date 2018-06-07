# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 19:15:36 2018

@author: 付学明
"""

import ExtractFeatures as ef

import numpy as np

class KNearestN:
    def __init__(self):
        pass
    
    def train(self,x,y):
        self.Xtr = x
        self.ytr = y
        
    def predict(self,X,k=5,dis=2):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test,dtype = self.ytr.dtype)
        for i in range(num_test):
            
            #距离
            if dis == 1:
                distances = np.sum(np.abs(self.Xtr-X[i,:]), axis = 1)
            elif dis == 2:
                distances = np.sqrt(np.sum((self.Xtr-X[i,:])**2, axis = 1))
            elif dis == 3:
                distances = np.power((np.sum((self.Xtr-X[i,:])**3, axis = 1)),1/3)
            else :
                distances = np.power((np.sum((self.Xtr-X[i,:])**dis, axis = 1)),1/dis)
            
            jishu = {}
            for j in range(k):
                min_index = np.argmin(distances)
                max_index = np.argmax(distances)
                if self.ytr[min_index] in jishu.keys():
                    jishu[self.ytr[min_index]] += 1
                else:
                    jishu[self.ytr[min_index]]=1
                distances[min_index] = distances[max_index]
            Ypred[i]=max(jishu.items(),key = lambda x: x[1])[0]
        #print(Ypred)
        return Ypred


feature = np.loadtxt('feature3.txt')
labels = np.loadtxt('labels3.txt')

picnum=567

#filename = '../gray/' + str(picnum) + '.jpg'
filename = '../image/' + str(picnum) + '.jpg'
test = ef.hist(filename)




#nn = KNearestN()0
#nn.train(feature,labels)
#t = nn.predict(test,k=50,dis=3)

'''
def DisCos(a,b):
    zi = np.sum(a*b,axis=1)
    mu = np.sqrt(np.sum(a**2,axis=1))*np.sqrt(np.sum(b**2,axis=1))
    
    return zi/mu
'''
def L2(a,b):
    distances = np.sqrt(np.sum((a-b)**2, axis = 1))
    return distances


#最近的前 K张图片
k=10

distance=L2(feature,test)
print(np.argsort(distance)[:k])

import matplotlib.pyplot as plt

from PIL import Image

img = Image.open('../image/'+str(picnum)+'.jpg')
#img = Image.open('../gray/'+str(picnum)+'.jpg')
img.show()

j=1
for i in np.argsort(distance)[:9]:
    filename = '../image/' + str(i) + '.jpg'
    #print(i)
    #filename = '../gray/' + str(i) + '.jpg'
    img=Image.open(filename)
    #img.show(i)
    plt.axis('off')
    plt.subplot(330+j)
    j+=1
    plt.imshow(img)
    #plt.imshow(img,cmap='gray')
    

