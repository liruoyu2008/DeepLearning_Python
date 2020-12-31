# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:26:08 2020

@author: Ryu
"""

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 载入数据集
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data[0])

# 数据归一化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
print(train_data[0])

"""
架设网络结构
"""
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', 
                  loss='mse', 
                  metrics=['mae'])
    return model

# K折验证
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = [] 
all_scores = []
for i in range(k):
     print('processing fold #', i)
     # 第i分区数据（用于验证）
     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
     # 其他所有分区数据（用于训练）
     partial_train_data = np.concatenate(
             [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
             axis=0)
     partial_train_targets = np.concatenate(
             [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
             axis=0)
     # 构建模型
     model = build_model()
     # 训练
     history = model.fit(partial_train_data, 
               partial_train_targets,
               epochs=num_epochs, 
               batch_size=1, 
               verbose=0    # 静默模式训练
               )
     
     # 记录训练趋势数据
     mae_history = history.history['val_mean_absolute_error']
     all_mae_histories.append(mae_history)
     average_mae_history = [np.mean([x[i] 
         for x in all_mae_histories]) for i in range(num_epochs)]
     
     # 验证
     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
     all_scores.append(val_mae)
     
     plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
     plt.xlabel('Epochs')
     plt.ylabel('Validation MAE')
     plt.show()
