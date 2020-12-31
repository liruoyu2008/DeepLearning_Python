# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:15:19 2020

@author: Ryu
"""

from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

# 载入数据集
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=10000)

"""
将多个行向量规整为二维矩阵
若原向量中出现过某数值，则已改数值为索引
二维矩阵中对应行内该索引处元素置为1
（意为忽略某单词出现的位置和次数，只关心该单词是否出现过）
"""
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# 将训练与测试数据矩阵化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 将训练与测试标签矩阵化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 网络结构
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# 训练
history = model.fit(x_train,y_train,epochs=20,batch_size=512)

# 输出训练历史
history_dict = history.history
loss_values = history_dict['loss']
epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'b',label = 'Training loss')

# 测试
test_loss, test_acc = model.evaluate(x_test, y_test)

# 输出
print('test_acc:', test_acc)