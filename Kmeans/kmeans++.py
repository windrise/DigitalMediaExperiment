# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     kmeans++
   Description :
   Author :       付学明
   date：          2018/5/30
-------------------------------------------------
   Change Activity:
                   2018/5/30:
-------------------------------------------------
__author__ = '付学明'
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def load_img(file='1.jpg'):
	filepath = 'image/'+file
	img = Image.open(filepath)
	a = np.array(img)
	return a


def plus(dataSet, k=2):
	lenth, dim = dataSet.shape
	_max = np.max(dataSet, axis=0)  # 线性映射最大值  axis=0列最大值
	_min = np.min(dataSet, axis=0)
	centers = []
	centers.append((_min + (_max - _min) * (np.random.rand(dim))))
	centers = np.array(centers)
	
	for i in range(1, k):
		distanceS = []
		for row in dataSet:
			distanceS.append(np.min(np.linalg.norm(row - centers, axis=1)))  # 计算离多个中心的距离里面最近的那个..
		temp = sum(distanceS) * np.random.rand()
		for j in range(lenth):
			temp -= distanceS[j]  # 依次剥离距离条
			if temp < 0:
				centers = np.append(centers, [dataSet[j]], axis=0)  # 保持0轴不塌陷
				break
	return centers


def kmeans(dataSet, k, maxIter=300):
	
	if k>=4 :
		centers = plus(dataSet, k)
	else:
		centers = random.sample(list(dataSet),k)
	labels = np.ones(len(dataSet))
	j = 0
	while 1 and j < maxIter:
		j += 1
		
		distance = np.zeros((len(dataSet), k))
		ii = 0
		for cent in centers:
			dis = np.sqrt(np.sum(((dataSet - cent)/255) ** 2, 1))
			distance[:, ii] = dis
			ii += 1
		label_new = np.argmin(distance, 1)
		# label_new = np.array(list(map(getLabel, dataSet)))
		#if sum(np.abs(labels - label_new)) == 0:  # 判断标签是否改变
		if sum(np.abs(labels -label_new))== 0:
			break
		labels = label_new
		for i in range(k):
			centers[i] = np.mean(dataSet[labels == i], axis=0)  # 更新聚类中心
		
		print("第{}次迭代,聚类中心".format(j))
		print(centers)
	return labels,centers


def kmeans_SSE(dataSet, k=16, maxIter=300):
	centers = plus(dataSet, k)
	'''
	def getLabel(data):
		distanceS = np.linalg.norm(data - centers, axis=1)  # 注意axis是等于1的
		
		return np.where(distanceS == np.min(distanceS))[0][0]
	
	'''
	labels = np.ones(len(dataSet))
	j = 0
	while 1 and j < maxIter:
		j += 1
		
		distance = np.zeros((len(dataSet),k))
		ii=0
		for cent in centers:
			dis = np.sqrt(np.sum(((dataSet - cent)/255)**2,1))
			distance[:,ii]=dis
			ii +=1
		label_new = np.argmin(distance,1)
		#label_new = np.array(list(map(getLabel, dataSet)))
		if sum(np.abs(labels - label_new)==0) == 0:  # 判断标签是否改变
			break
		labels = label_new
		for i in range(k):
			centers[i] = np.mean(dataSet[labels == i], axis=0)  # 更新聚类中心
			
		print("第{}次迭代,聚类中心".format(j))
		print(centers)
	SSE = sum([sum([(j - centers[i]).dot(j - centers[i]) for j in dataSet[labels == i]]) for i in range(k)])
	return SSE

pic='35.jpg'
data = load_img(pic)
X = data.reshape(-1,3)


k=np.array(range(2,1000))
SSE = np.zeros(len(k))
for i in range(len(k)):
	print(i)
	SSE[i] = kmeans_SSE(X,k=i+2)

plt.figure(1)
plt.plot(k,SSE/1000000)
plt.show()
'''
labels, centers = kmeans(X,k=10)
result = X
for i in range(len(X)):
	result[i, :] = centers[int(labels[i])]
result = result.reshape(data.shape)
image=Image.fromarray(result)
#image.save('kmeans.jpg')
image.show()
#plt.imshow(image)
#plt.show()
'''
