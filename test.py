import os

# tf2.1下测试程序是否再GPU上运行
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
print(tf.test.is_built_with_cuda())