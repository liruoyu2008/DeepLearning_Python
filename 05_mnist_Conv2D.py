# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.engine import network
from keras.layers.core import Activation
from keras.utils import to_categorical

# 准备训练与测试数据（将原始数据整合为4D张量，形式采用tf的颜色通道轴后置式）
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# 准备标签（独热变换、独热编码，通过位指示器形式表示标签）
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 添加卷积层和池化层，对输入数据进行特征卷积
network = models.Sequential()
network.add(layers.Conv2D(
    32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 将最终得到的3D特征图（多张2D特征图的组合）展平为向量，输入到普通的密集连接神经网络
network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

# 打印网络概况
print(network.summary())

# 编译网络
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 开始训练并输出训练数据
network.fit(train_images, train_labels, epochs=5, batch_size=64)

# 测试并输出测试过程
test_loss, test_acc = network.evaluate(test_images, test_labels)

# 打印测试精度
print('test_acc:', test_acc)
model = models
