# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 11:28:22 2020

@author: Ryu
"""

from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# 载入数据集并输出样本轴大小
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)
print(len(train_data), len(test_data))

# 输出解码后的示例
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])
decoded_newswire = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_newswire)


def vectorize_sequences(sequences, dimension=10000):
    """
    定义数据向量化的函数,
    向量化后的数据表示样本内是否出现了该单词
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# 向量化数据
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 向量化标签（对应后面损失函数的选择）
# 如果标签集通过one-hot编码为矩阵，例如[[1,0,0],[0,0,1],[0,1,0]]，则对应损失函数为categorical_crossentropy
# 如果保持为原始标签向量集,例如[1,3,2]，则对应损失函数为sparse_categorical_crossentropy
one_hot_train_labels = np.array(train_labels)
one_hot_test_labels = np.array(test_labels)
#one_hot_train_labels = to_categorical(train_labels)
#one_hot_test_labels = to_categorical(test_labels)

# 将训练集分为两部分（分别用于测试、评估）
# 测试集则作为全新数据进行评估或测试
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 网络结构
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# 编译(损失函数的选择需匹配标签集的表示形式)
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              # loss='categorical_crossentropy',
              metrics=['accuracy'])


# 训练
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 绘制训练损失和验证损失
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 评估
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

# 预测
predicaitons = model.predict(x_test)
print(np.argmax(predicaitons[0]))
