# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     ExtractFeatures
   Description :
   提取图片数据库中所有图片的HSV模型的二维颜色直方图特征，并存储
   
   
   Author :       付学明
   date：          2018/6/5
-------------------------------------------------
   Change Activity:
                   2018/6/5:
-------------------------------------------------
__author__ = '付学明'
"""
import cv2
import numpy as np
from skimage import exposure



def hist(filename='../gray/1.jpg'):

    img = cv2.imread(filename)
    '''
    HSV 颜色空间
    channels=[0,1] 同时提取 H，S两个通道
    bins=[180,256] 每个通道的特征子区间数 H，S通道分别为180，256
    range=[0， 180， 0， 256]H 的取值范围在 0 到 180， S 的取值范围在 0 到 256
    '''
    #彩色图
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0,256])
    #hist = cv2.calcHist([img],[0],None,[256],[0,255])
    
    #hist_falt=hist.flatten()
    
    #直方图均衡化
    #img1 = exposure.equalize_hist(hist)
    img1=hist
    #plt.imshow(img1)
    arr1 = img1.flatten()
    #plt.hist(arr1, bins=256, normed=1,edgecolor='None',facecolor='red')
    
    return arr1.reshape((1,-1))



if __name__ == '__main__':
    
    #feature = np.zeros((1000,180*256))
    feature = np.zeros((1000,256))
    labels = np.zeros(1000)
    for i in range(1000):
        #filename = '../gray/'+str(i)+'.jpg'
        filename = '../image/'+str(i)+'.jpg'
        arr = hist(filename)
        feature[i,:]=arr
        labels[i]=i//100
    np.savetxt('feature3.txt',feature,fmt='%0.7f')
    np.savetxt('labels3.txt',labels)
'''
0  180x256  均衡化
1  45x64  均衡化
2  45x64  非均衡化
3  180x256  非均衡化
4   256    灰度图
'''

