import tensorflow as tf
import os
import matplotlib.pyplot as plt


def show_history(history):
    """展示网络训练过程

    Args:
        history ([type]): 训练的历史数据
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # tf2.1下测试程序是否再GPU上运行
    print(tf.config.list_physical_devices('GPU'))
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    print(tf.test.is_built_with_cuda())
