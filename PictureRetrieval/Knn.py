# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 19:31:52 2018

@author: 付学明
"""

import numpy as np    


class KNearestN:
    def __init__(self):
        pass
    
    def train(self,x,y):
        self.Xtr = x
        self.ytr = y
    def predict(self,X,k=1,dis=2):
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
    
'''   
    def predict(self, X, k=1,dis = 2):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        for i in range(num_test):
            if dis == 1:
                distances = np.sum(np.abs(self.Xtr-X[i,:]), axis = 1)
            elif dis == 2:
                distances = np.sqrt(np.sum((self.Xtr-X[i,:])**2, axis = 1))
            elif dis == 3:
                distances = np.power((np.sum((self.Xtr-X[i,:])**3, axis = 1)),1/3)
            else :
                distances = np.power((np.sum((self.Xtr-X[i,:])**dis, axis = 1)),1/dis)
                
        
        closest_y = y_train[np.argsort(distances)[:k]]
        u, indices = np.unique(closest_y, return_inverse=True)
        Ypred[i] = u[np.argmax(np.bincount(indices))]
        return Ypred
'''


#训练数据确定K值
feature = np.loadtxt('feature.txt')
labels = np.loadtxt('labels.txt')

data = np.column_stack((feature,labels))
m = feature.shape[1]


# 数据集划分
indices = np.array(range(10))
np.random.shuffle(indices)


index1 = np.array(range(0,10))
np.random.shuffle(index1)

index2 = np.array(range(0,100))
np.random.shuffle(index2)


for j in range(10):
    for i in index1:
        indice = index2[10*j:10*(j+1)]+ 100*i
        np.random.shuffle(indice)
        indices = np.row_stack((indices,indice))
indices=indices[1:,:]
indices=indices.T
'''
for i in range(1,10):
    indice = np.array(range(100))+100*i
    np.random.shuffle(indice)
    indices = np.row_stack((indices,indice))

'''

x_train = data[indices[0,:80],:m]
y_train = data[indices[0,:80],m]

x_test = data[indices[0,80:],:m]
y_test = data[indices[0,80:],m]



for i in range(1,10):
    
    x_train = np.concatenate((x_train,data[indices[i,:80],:m]),axis=0)
    y_train = np.concatenate((y_train,data[indices[i,:80],m]),axis=0)
    x_test = np.concatenate((x_test,data[indices[i,80:],:m]),axis=0)
    y_test = np.concatenate((y_test,data[indices[i,80:],m]),axis=0)


knn = KNearestN()
knn.train(x_train,y_train)

'''
k_choice=np.array(range(1,100))
acc= np.zeros(len(k_choice))

i=0
for kval in k_choice:
    y_predict = knn.predict(x_test, k=kval)
    acc[i] = np.mean(y_predict == y_test)
    i +=1
    
print('accuracy :')
print(acc)

print('The best K:')
print(np.argmax(acc))


'''
#非均衡化  采样通道较小
#比较合适的K为  8

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

x_train_folds = np.array_split(x_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

for k_val in k_choices:
    print('k = ' + str(k_val))
    k_to_accuracies[k_val] = []
    for i in range(num_folds):
        x_train_cycle = np.concatenate([f for j,f in enumerate (x_train_folds) if j!=i])
        y_train_cycle = np.concatenate([f for j,f in enumerate (y_train_folds) if j!=i])
        x_val_cycle = x_train_folds[i]
        y_val_cycle = y_train_folds[i]
        knn = KNearestN()
        knn.train(x_train_cycle, y_train_cycle)
        y_val_pred = knn.predict(x_val_cycle, k_val)
        acc=np.mean(y_val_cycle == y_val_pred)
        k_to_accuracies[k_val].append(acc)
        print(acc)


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
