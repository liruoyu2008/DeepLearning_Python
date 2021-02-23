# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import ipykernel

# 准备训练集与测试集（整形、归一）
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 准备标签（独热变换、独热编码，通过位指示器形式表示标签）
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 添加全连接层和 softmax 分类器，指定优化器、损失函数和训练指标并编译网络
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

# 开始训练并输出训练数据
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 测试并输出测试过程
test_loss, test_acc = network.evaluate(test_images, test_labels)
 
# 打印测试精度
print('test_acc:', test_acc)